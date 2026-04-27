//! Qwen3 / Qwen3-MoE tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble text<tool_call>{"name": "x", "arguments": {...}}</tool_call>
//! more text<tool_call>{"name": "y", "arguments": {...}}</tool_call>
//! ```
//!
//! Each `<tool_call>...</tool_call>` block carries a single JSON object
//! with at minimum a `name` field; arguments arrive under either
//! `"arguments"` or `"parameters"` (the chat templates in the wild use
//! both).
//!
//! # Strict gating
//!
//! The historical `parse_qwen3_tool_calls` in `helpers/tool_calls.rs`
//! also accepted bare JSON without a sentinel wrapper as a fallback.
//! This implementation deliberately drops that fallback: a model output
//! that contains a JSON object resembling `{"name": "x", ...}` but
//! without the `<tool_call>` wrapper is plain text, not a tool call.
//!
//! # Per-call atomic emission
//!
//! See [`super::json_sentinel`] for the shared state machine — Qwen3
//! and SmolLM2 both delegate to it because they share the same wire
//! format.
//!
//! # Residual false-positive risk
//!
//! When tools are bound, the parser cannot distinguish a model that
//! genuinely wants to call a tool from a model that is illustrating a
//! tool call inside a Markdown code fence (e.g. answering "show me an
//! example tool call"). Schema validation reduces blast radius — random
//! model output rarely satisfies a typed schema — but does not
//! eliminate the class. The structural fix lives upstream in the model
//! / tokenizer (sandboxed special tokens that user text cannot
//! produce). When tools are NOT bound,
//! [`ChatTurn::make_parser`](crate::runtime::chat::ChatTurn::make_parser)
//! returns the passthrough parser and this protocol is never
//! instantiated, so the no-tools case is structurally clean.

// All production wiring (sentinel strings, codec choice, render_tools,
// prepare_messages) lives in `protocol.rs`'s `QWEN3` registry entry —
// this module is purely test scaffolding now. The Qwen3 dialect is
// `<tool_call>JSON</tool_call>` with the Hermes-permissive codec
// (empty `[]` → zero calls, OpenAI spec-shape echo peeled).

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::runtime::chat::{IncrementalToolCallParser, ToolDirectory};

/// Test-only constructor: dispatches through the registry so the
/// tests exercise the same parser path the gateway will use.
/// Crate-public so the wire-mapper tests can build a Qwen3 parser
/// without duplicating the registry lookup.
#[cfg(test)]
pub(crate) fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    use crate::runtime::chat::tool_protocol_for;
    tool_protocol_for("Qwen3ForCausalLM", &serde_json::Value::Null)
        .expect("Qwen3 arch is registered")
        .make_parser(directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::protocol_test_kit::{
        add_tool, directory_with_add, last_stop_reason, run,
    };
    use crate::runtime::chat::{
        DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory, ToolSpec,
    };
    use serde_json::json;

    use crate::runtime::chat::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;



    #[test]
    fn valid_call_emits_start_args_end() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { index: 0, name } if name == "add"));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { index: 0, .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { index: 0, .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    /// The streaming-claim test: a single call must be fully emitted
    /// when its closing sentinel arrives, before any `finish()` call.
    /// This proves per-call streaming (not just end-of-stream batch).
    #[test]
    fn call_emitted_atomically_when_close_sentinel_arrives() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        // Start, ArgsDelta, End — all here, before finish().
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::Stop { .. })));
    }






    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: bare JSON resembling a tool call must NOT be parsed.
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains('<'),
            _ => true,
        }));
        events.extend(p.feed(r#"all>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> done"#));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "preamble "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " done"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_close_sentinel_split_across_feeds() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let mut events = Vec::new();
        events.extend(p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_"#));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn sentinel_prefix_that_doesnt_resolve_emits_as_text() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let mut events = Vec::new();
        events.extend(p.feed("hello <to"));
        events.extend(p.feed("morrow"));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "hello <tomorrow");
    }

    #[test]
    fn utf8_multibyte_split_across_feeds_does_not_panic() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let mut events = Vec::new();
        events.extend(p.feed("héllo "));
        events.extend(p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "héllo ");
    }



    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":1"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::Unterminated,
                ..
            }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_directory_makes_every_call_terminal_unknown() {
        let dir = Arc::new(ToolDirectory::new(vec![]).unwrap());
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","arguments":{}}</tool_call>"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "add"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }



    #[test]
    fn payload_over_limit_without_close_is_terminal() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<tool_call>"));
        events.extend(p.feed(&oversize));
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(
            source,
            ParserError::PayloadTooLarge { limit_bytes }
                if *limit_bytes == MAX_TOOL_CALL_PAYLOAD_BYTES
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn payload_over_limit_with_close_in_same_feed_still_fatal() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let mut chunk = String::from("<tool_call>");
        chunk.push_str(&"x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1));
        chunk.push_str("</tool_call>");
        let events = p.feed(&chunk);
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::PayloadTooLarge { .. },
                ..
            }
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::ToolCallStart { .. })));
    }

    #[test]
    fn error_message_does_not_include_oversized_payload() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let secret = "SUPER_SECRET_TOKEN_THAT_SHOULD_NOT_LEAK";
        let mut chunk = String::from("<tool_call>");
        chunk.push_str(secret);
        chunk.push_str(&"x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES));
        let events = p.feed(&chunk);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError");
        };
        let msg = source.to_string();
        assert!(
            !msg.contains(secret),
            "oversized payload error message must not include payload bytes; got: {msg}"
        );
    }
}
