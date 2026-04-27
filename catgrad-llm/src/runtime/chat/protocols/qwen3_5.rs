//! Qwen3.5 tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble<tool_call>
//! <function=tool_name>
//! <parameter=arg_name>arg_value</parameter>
//! <parameter=other_arg>another</parameter>
//! </function>
//! </tool_call>postamble
//! ```
//!
//! The outer sentinel pair is the same as Qwen3 (`<tool_call>` /
//! `</tool_call>`), but the payload is XML-shaped rather than JSON: a
//! single `<function=NAME>...</function>` block whose body is a sequence
//! of `<parameter=KEY>VALUE</parameter>` blocks.
//!
//! Per-parameter `VALUE` is parsed with `serde_json::from_str` after
//! trimming; failures fall back to a bare string. This mirrors the
//! `parse_scalar` helper in the legacy non-streaming parser
//! (`helpers/tool_calls.rs`).
//!
//! Multiple parallel calls per generation are emitted as multiple
//! back-to-back `<tool_call>...</tool_call>` blocks (NOT multiple
//! functions inside one block); each block is one call.
//!
//! # Strict gating
//!
//! As with Qwen3: a model output that contains XML-shaped text but no
//! `<tool_call>` wrapper is plain text, never a tool call.
//!
//! # Per-call atomic emission
//!
//! Same contract as Qwen3: `<tool_call>` opens a buffer; only when
//! `</tool_call>` arrives do we parse, validate, and emit
//! `ToolCallStart` + `ToolCallArgsDelta` + `ToolCallEnd` as one atomic
//! triple. Validation failures take the `UnknownTool` / `InvalidArgs` /
//! `ParseError` paths and never emit a `ToolCallStart`.

// All production wiring lives in `protocol.rs`'s registry entry.
// Wire dialect: `<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>`
// — XML payload between Hermes-style sentinels. Parsed by
// `codecs::XmlFunctionCodec` via the shared `SentinelEngine`.

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::runtime::chat::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;
#[cfg(test)]
use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory,
};

#[cfg(test)]
fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    use crate::runtime::chat::tool_protocol_for;
    tool_protocol_for("Qwen3_5ForCausalLM", &serde_json::Value::Null)
        .expect("arch is registered")
        .make_parser(directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::protocol_test_kit::{
        add_tool, directory_with_add, last_stop_reason, run,
    };
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;


    fn calculator_tool() -> ToolSpec {
        ToolSpec::new(
            "calculator",
            Some("calculate".into()),
            json!({
                "type": "object",
                "properties": {
                    "lhs": { "type": "number" },
                    "rhs": { "type": "number" },
                    "op":  { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                },
                "required": ["lhs", "rhs", "op"],
                "additionalProperties": false,
            }),
        )
    }

    fn greet_tool() -> ToolSpec {
        ToolSpec::new(
            "greet",
            None,
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "times": { "type": "number" },
                },
                "required": ["name"],
                "additionalProperties": false,
            }),
        )
    }

    fn directory_with_calculator() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![calculator_tool()]).unwrap())
    }

    fn directory_with_greet() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![greet_tool()]).unwrap())
    }


    #[test]
    fn valid_call_emits_start_args_end() {
        let mut p = make_parser(directory_with_greet());
        let events = run(
            &mut *p,
            &[
                "<tool_call>\n<function=greet>\n<parameter=name>\"Alice\"</parameter>\n</function>\n</tool_call>",
            ],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "greet"
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::ToolCallArgsDelta { index: 0, .. }
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2])
        };
        assert_eq!(args, &json!({ "name": "Alice" }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }











    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["<tool_call><function=add><parameter=a>1</parameter>"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}")
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        // Should not have emitted the partial sentinel as text.
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains("<tool_c"),
            _ => true,
        }));
        events.extend(p.feed(
            "all><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> done",
        ));
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
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_",
        ));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_xml_without_sentinel_is_plain_text() {
        // Critical: bare XML resembling a tool call must NOT be parsed.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "Here is some XML: <function=add><parameter=a>1</parameter></function>",
            ],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "Here is some XML: <function=add><parameter=a>1</parameter></function>"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }



    // Codec helper tests (parse_scalar / parse_function_block)
    // moved into `codecs::xml_function` — they belong with the
    // implementation, not the protocol-specific module.
}
