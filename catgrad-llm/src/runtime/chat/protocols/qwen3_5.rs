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
    fn valid_call_with_multiple_parameters() {
        // String + number args; confirms parse_scalar's JSON-promotion of
        // numeric and quoted-string values.
        let mut p = make_parser(directory_with_calculator());
        let events = run(
            &mut *p,
            &[
                "<tool_call>\n<function=calculator>\n<parameter=lhs>1353785</parameter>\n<parameter=rhs>790489</parameter>\n<parameter=op>\"div\"</parameter>\n</function>\n</tool_call>",
            ],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["lhs"], json!(1353785));
        assert_eq!(args["rhs"], json!(790489));
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameter_value_falls_back_to_bare_string() {
        // Bare `div` (no quotes) — JSON parse fails, fall back to string.
        let mut p = make_parser(directory_with_calculator());
        let events = run(
            &mut *p,
            &[
                "<tool_call>\n<function=calculator>\n<parameter=lhs>1</parameter>\n<parameter=rhs>2</parameter>\n<parameter=op>div</parameter>\n</function>\n</tool_call>",
            ],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn multiple_calls_in_sequence() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "<tool_call>\n<function=add>\n<parameter=a>1</parameter>\n<parameter=b>2</parameter>\n</function>\n</tool_call>",
                "<tool_call>\n<function=add>\n<parameter=a>3</parameter>\n<parameter=b>4</parameter>\n</function>\n</tool_call>",
            ],
        );
        // Start(0), ArgsDelta(0), End(0), Start(1), ArgsDelta(1), End(1), Stop
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, name } if name == "add"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn text_then_call_then_text_in_single_feed() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "first <tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> last",
            ],
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
    fn malformed_parameter_tag_is_terminal_with_protocol_error() {
        // <parameter=a missing its closing '>' — this hits the
        // "unterminated <parameter=...> tag" branch.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "<tool_call><function=add><parameter=a 1</parameter></function></tool_call>",
            ],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_function_block_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["<tool_call>just some text without a function block</tool_call>"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["<tool_call></tool_call>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
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

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across arbitrary boundaries produces the same final
    //! decoded turn (or the same DecodeFailure).

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // single valid call
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>",
            // call with newlines (the canonical chat-template shape)
            "<tool_call>\n<function=add>\n<parameter=a>1</parameter>\n<parameter=b>2</parameter>\n</function>\n</tool_call>",
            // call with surrounding text
            "prefix <tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> suffix",
            // two calls in sequence
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call><tool_call><function=add><parameter=a>3</parameter><parameter=b>4</parameter></function></tool_call>",
            // sentinel-shaped text that isn't a sentinel
            "the docs say <tool_call but it's just text",
            // unknown tool (fatal — chunk-invariance still holds)
            "<tool_call><function=missing></function></tool_call>",
            // bare-string parameter value falling back through parse_scalar
            "<tool_call><function=add><parameter=a>not_json</parameter><parameter=b>2</parameter></function></tool_call>",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..8,
            split in 0_usize..400,
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &[split]);
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }

        #[test]
        fn n_way_split_is_invariant(
            input_idx in 0_usize..8,
            mut splits in prop::collection::vec(0_usize..400, 1..5),
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
