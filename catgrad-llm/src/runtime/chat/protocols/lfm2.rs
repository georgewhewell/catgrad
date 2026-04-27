//! LFM2 / LFM2.5 tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble<|tool_call_start|>[calculator(lhs=1, rhs=2, op="div")]<|tool_call_end|>postamble
//! ```
//!
//! The block between `<|tool_call_start|>` and `<|tool_call_end|>` may
//! carry one of two payload shapes:
//!
//! - **Pythonic** (LiquidAI default): a Python list literal of calls,
//!   `[name1(k=v, ...), name2(k=v, ...)]`. The outer `[...]` is also
//!   accepted as omitted (a single bare call).
//! - **JSON** (alternate, requested by system prompt): `[{"name": "x",
//!   "arguments": {...}}]` — same shape as Qwen3 but wrapped in a list,
//!   and a bare object outside the list is also accepted.
//!
//! The dispatch is by sniffing the first non-whitespace byte: `[` or `{`
//! routes to JSON; anything else routes to Pythonic. If JSON parsing
//! fails on a payload that opened with `[` or `{`, the parser falls back
//! to Pythonic — model outputs occasionally mix the two (e.g. a JSON-
//! looking dict literal that uses single quotes).
//!
//! # Multiple calls per block
//!
//! Both payload shapes can carry multiple calls. The parser emits one
//! `ToolCallStart` / `ToolCallArgsDelta` / `ToolCallEnd` triple per
//! parsed call, with strictly increasing indices, atomically (no
//! interleaving). If call N+1 fails validation, calls 0..N have already
//! been emitted and the fatal event terminates the parser.
//!
//! # References
//!
//! - LFM2.5 chat template:
//!   <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/chat_template.jinja>
//! - SGLang reference parser:
//!   <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/lfm2_detector.py>

// All production wiring lives in `protocol.rs`'s `LFM2` registry
// entry. The dialect is `<|tool_call_start|>...<|tool_call_end|>`
// containing either Pythonic call-list (`[name(k=v, ...), ...]`) or
// JSON (`[{...}]`). The sentinel-engine is configured with a
// `MultiCodec` that tries JSON first, then Pythonic — matches the
// reference SGLang `lfm2_tool_parser.py` byte-sniff dispatcher.

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
    tool_protocol_for("Lfm2ForCausalLM", &serde_json::Value::Null)
        .expect("Lfm2 arch is registered")
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

    fn directory_with_calculator() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![calculator_tool()]).unwrap())
    }


    #[test]
    fn pythonic_call_with_outer_list() {
        let mut p = make_parser(directory_with_calculator());
        let events = run(
            &mut *p,
            &[
                "<|tool_call_start|>[calculator(lhs=1353785, rhs=790489, op=\"div\")]<|tool_call_end|>",
            ],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "calculator"
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args["lhs"], json!(1353785));
        assert_eq!(args["rhs"], json!(790489));
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn pythonic_call_without_outer_list() {
        // Bare `name(args)` (no enclosing `[...]`) is also accepted.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["<|tool_call_start|>add(a=1, b=2)<|tool_call_end|>"],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!()
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn pythonic_single_quoted_string_arg() {
        let mut p = make_parser(directory_with_calculator());
        let events = run(
            &mut *p,
            &["<|tool_call_start|>[calculator(lhs=1, rhs=2, op='div')]<|tool_call_end|>"],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["op"], json!("div"));
    }

    #[test]
    fn json_call_array_payload() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}}]<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!()
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn json_call_bare_object_payload() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<|tool_call_start|>{"name":"add","arguments":{"a":1,"b":2}}<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn json_call_parameters_key_accepted() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<|tool_call_start|>[{"name":"add","parameters":{"a":1,"b":2}}]<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
    }

    #[test]
    fn pythonic_multiple_calls_emit_sequential_indices() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "<|tool_call_start|>[add(a=1,b=2), add(a=3,b=4)]<|tool_call_end|>",
            ],
        );
        // Two triples + Stop = 7 events.
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn json_multiple_calls_emit_sequential_indices() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]<|tool_call_end|>"#,
            ],
        );
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
    }

    #[test]
    fn calls_in_separate_blocks_keep_index_sequential() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "<|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|>",
                "<|tool_call_start|>[add(a=3,b=4)]<|tool_call_end|>",
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
    }

    #[test]
    fn text_then_call_then_text_in_single_feed() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "first <|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|> last",
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "first "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " last"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }



    #[test]
    fn second_call_unknown_after_first_succeeds_emits_first_then_fatal() {
        // Validates the partial-then-fatal contract: calls 0..N validated,
        // call N+1 is unknown — events must include 0..N's full triples
        // before the UnknownTool/Stop pair.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                "<|tool_call_start|>[add(a=1,b=2), delete_db()]<|tool_call_end|>",
            ],
        );
        // Triple for add + UnknownTool + Stop = 5 events.
        assert_eq!(events.len(), 5);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(
            &events[2],
            DecodeEvent::ToolCallEnd { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }


    #[test]
    fn empty_payload_is_terminal() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["<|tool_call_start|><|tool_call_end|>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_block_at_eos_is_terminal() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["<|tool_call_start|>[add(a=1,b=2)"]);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError")
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_pythonic_without_sentinel_is_plain_text() {
        // No tool-call sentinel = no parsing, even if the text looks
        // exactly like a Pythonic call. Matches the Qwen3 strict-gating
        // rule.
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["here is some text: add(a=1, b=2)"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "here is some text: add(a=1, b=2)"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <|tool_c"));
        events.extend(p.feed("all_start|>[add(a=1,b=2)]<|tool_call_end|> done"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "preamble "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " done"
        ));
    }

    #[test]
    fn partial_close_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("<|tool_call_start|>[add(a=1,b=2)]<|tool_call_"));
        assert!(events.is_empty(), "got events early: {events:?}");
        events.extend(p.feed("end|>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }



    // Codec helper tests (parse_python_call / split_top_level / etc.)
    // moved into `codecs::pythonic` — they belong with the
    // implementation. The behaviour is exercised end-to-end here via
    // the engine.
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
            // Pythonic with outer list
            "<|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|>",
            // Pythonic without outer list
            "<|tool_call_start|>add(a=1,b=2)<|tool_call_end|>",
            // JSON array form
            r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}}]<|tool_call_end|>"#,
            // JSON bare object form
            r#"<|tool_call_start|>{"name":"add","arguments":{"a":1,"b":2}}<|tool_call_end|>"#,
            // Two pythonic calls in one block
            "<|tool_call_start|>[add(a=1,b=2), add(a=3,b=4)]<|tool_call_end|>",
            // Sentinel-shaped text that isn't a sentinel
            "the docs say <|tool_call_start but it's just text",
            // Call wrapped in surrounding text
            "prefix <|tool_call_start|>add(a=1,b=2)<|tool_call_end|> suffix",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..8,
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
            input_idx in 0_usize..8,
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
