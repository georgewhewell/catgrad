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

use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::runtime::chat::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use crate::types;

use super::json_sentinel;

pub(super) const TOOL_CALL_OPEN: &str = "<tool_call>";
pub(super) const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Construct a Qwen3 parser bound to the given tool directory.
///
/// The parser owns the `Arc<ToolDirectory>`, so the returned
/// `Box<dyn IncrementalToolCallParser>` is `'static`.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    json_sentinel::make_parser(directory, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
}

/// Render the bound tool list into the JSON shape the Qwen3 chat
/// templates expect — the OpenAI-style `[{"type": "function",
/// "function": {...}}, ...]` envelope. The chat template iterates over
/// `tools` and reads `tool.function.name`, `tool.function.description`,
/// `tool.function.parameters`.
pub fn render_tools(specs: &[ToolSpec]) -> JsonValue {
    json_sentinel::render_openai_tool_envelope(specs)
}

/// Qwen3's chat template natively renders the tool list, so the
/// protocol does not need to inject any system-prompt scaffolding.
/// Identity over the message list.
pub fn prepare_messages(_specs: &[ToolSpec], messages: Vec<types::Message>) -> Vec<types::Message> {
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{
        DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory, ToolSpec,
    };
    use serde_json::json;

    use super::super::json_sentinel::MAX_TOOL_CALL_PAYLOAD_BYTES;

    fn add_tool() -> ToolSpec {
        ToolSpec::new(
            "add",
            Some("add two numbers".into()),
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
                "additionalProperties": false,
            }),
        )
    }

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
    }

    fn run(parser: &mut dyn IncrementalToolCallParser, chunks: &[&str]) -> Vec<DecodeEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.feed(chunk));
        }
        events.extend(parser.finish(StopReason::EndOfText));
        events
    }

    fn last_stop_reason(events: &[DecodeEvent]) -> StopReason {
        events
            .iter()
            .rev()
            .find_map(|e| match e {
                DecodeEvent::Stop { reason } => Some(*reason),
                _ => None,
            })
            .expect("expected a Stop event")
    }

    #[test]
    fn plain_text_passes_through_as_text_delta() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(&mut *p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

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
    fn unknown_tool_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"delete_db","arguments":{}}</tool_call>"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":"one"}}</tool_call>"#],
        );
        assert_eq!(events.len(), 2);
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(&mut *p, &["<tool_call>not json at all</tool_call>"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_name_field_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(&mut *p, &[r#"<tool_call>{"arguments":{}}</tool_call>"#]);
        assert_eq!(events.len(), 2);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::MissingField("name")));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn parameters_key_is_accepted_as_arguments() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","parameters":{"a":1,"b":2}}</tool_call>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
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
    fn multiple_valid_tool_calls_in_sequence() {
        let mul = ToolSpec::new(
            "mul",
            None,
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
            }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![add_tool(), mul]).unwrap());
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[
                r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
                r#"<tool_call>{"name":"mul","arguments":{"a":3,"b":4}}</tool_call>"#,
            ],
        );
        // Start(0,add), ArgsDelta(0), End(0), Start(1,mul), ArgsDelta(1), End(1), Stop
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, name } if name == "mul"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn text_then_call_then_text_in_single_feed() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"first <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> last"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "first "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " last"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
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
    fn empty_payload_is_terminal_parse_error() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        let events = run(&mut *p, &["<tool_call></tool_call>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn after_fatal_error_subsequent_feed_and_finish_return_empty() {
        let dir = directory_with_add();
        let mut p = make_parser(dir);
        // Trigger fatal via unknown tool.
        let first = p.feed(r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#);
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &first[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        let after_feed = p.feed("any further text");
        assert!(after_feed.is_empty(), "got events: {after_feed:?}");
        let after_more =
            p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        assert!(after_more.is_empty(), "got events: {after_more:?}");
        let after_finish = p.finish(StopReason::EndOfText);
        assert!(after_finish.is_empty(), "got events: {after_finish:?}");
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

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across chunk boundaries produces the same final
    //! decoded turn (or the same DecodeFailure).

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            "hello world",
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
            r#"prefix <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> suffix"#,
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call><tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>"#,
            "the docs say <tool_call> but it's just text",
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> done"#,
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..7,
            split in 0_usize..200,
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &[split]);
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }

        #[test]
        fn n_way_split_is_invariant(
            input_idx in 0_usize..7,
            mut splits in prop::collection::vec(0_usize..200, 1..5),
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            splits.sort_unstable();
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &splits);
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }
    }
}
