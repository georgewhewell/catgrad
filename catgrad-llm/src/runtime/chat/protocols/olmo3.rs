//! OLMo 3 tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble<function_calls>[calculator(lhs=1, rhs=2, op="div"), other(arg=1)]</function_calls>postamble
//! ```
//!
//! The block between `<function_calls>` and `</function_calls>` carries
//! a Pythonic payload: a Python list literal of zero-or-more calls,
//! `[name1(k=v, ...), name2(k=v, ...)]`. The outer `[...]` is also
//! accepted as omitted (a single bare call), matching the chat
//! template's modern path which renders a sequence of bare
//! `name(args)` expressions.
//!
//! # Multiple calls per block
//!
//! Multiple calls in one block emit one `ToolCallStart` /
//! `ToolCallArgsDelta` / `ToolCallEnd` triple per parsed call, with
//! strictly increasing indices, atomically (no interleaving). If call
//! N+1 fails validation, calls 0..N have already been emitted and the
//! fatal event terminates the parser (mirrors LFM2).
//!
//! # References
//!
//! - OLMo-3 chat template:
//!   <https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/chat_template.jinja>

use std::sync::Arc;


use crate::runtime::chat::codecs::PythonicCallsCodec;
use crate::runtime::chat::sentinel_engine::{MAX_TOOL_CALL_PAYLOAD_BYTES, SentinelEngine};
use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory,
};

const TOOL_CALL_OPEN: &str = "<function_calls>";
const TOOL_CALL_CLOSE: &str = "</function_calls>";

/// Construct an OLMo 3 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Olmo3Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the OLMo-3 chat
/// template expects.
///

struct Olmo3Parser(SentinelEngine);

impl Olmo3Parser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self(SentinelEngine::new_pair(
            directory,
            Box::new(PythonicCallsCodec),
            TOOL_CALL_OPEN,
            TOOL_CALL_CLOSE,
        ))
    }
}

impl IncrementalToolCallParser for Olmo3Parser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        self.0.feed(text)
    }
    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        self.0.finish(reason)
    }
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
        let mut p = Olmo3Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[
                "<function_calls>[calculator(lhs=1353785, rhs=790489, op=\"div\")]</function_calls>",
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
        // Bare `name(args)` (no enclosing `[...]`) is also accepted —
        // matches the chat template's modern path which renders bare
        // `name(args)` per call.
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<function_calls>add(a=1, b=2)</function_calls>"],
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
        let mut p = Olmo3Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &["<function_calls>[calculator(lhs=1, rhs=2, op='div')]</function_calls>"],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["op"], json!("div"));
    }

    #[test]
    fn multiple_calls_in_sequence_emit_sequential_indices() {
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<function_calls>[add(a=1,b=2), add(a=3,b=4)]</function_calls>"],
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
    fn multiple_blocks_keep_indices_sequential() {
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<function_calls>[add(a=1,b=2)]</function_calls>",
                "<function_calls>[add(a=3,b=4)]</function_calls>",
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
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["first <function_calls>[add(a=1,b=2)]</function_calls> last"],
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
    fn empty_payload_is_terminal() {
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(&mut p, &["<function_calls></function_calls>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_block_at_eos_is_terminal() {
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(&mut p, &["<function_calls>[add(a=1,b=2)"]);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError")
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_payload_without_sentinel_is_plain_text() {
        // No tool-call sentinel = no parsing, even if the text looks
        // exactly like a Pythonic call. Matches the strict-gating rule.
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(&mut p, &["here is some text: add(a=1, b=2)"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "here is some text: add(a=1, b=2)"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = Olmo3Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <function_c"));
        events.extend(p.feed("alls>[add(a=1,b=2)]</function_calls> done"));
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
        let mut p = Olmo3Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("<function_calls>[add(a=1,b=2)]</function_"));
        assert!(events.is_empty(), "got events early: {events:?}");
        events.extend(p.feed("calls>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }



    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // Validates the partial-then-fatal contract: calls 0..N
        // validated, call N+1 is unknown — events must include 0..N's
        // full triples before the UnknownTool/Stop pair.
        let mut p = Olmo3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<function_calls>[add(a=1,b=2), delete_db()]</function_calls>"],
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

    // Codec-helper unit tests (parse_python_call / parse_python_value /
    // split_top_level / find_top_level_char) moved into the
    // `codecs::pythonic` module — they test the codec, not the OLMo-3
    // wire frame, so they live with the implementation.

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
            "<function_calls>[add(a=1,b=2)]</function_calls>",
            // Pythonic without outer list
            "<function_calls>add(a=1,b=2)</function_calls>",
            // Two pythonic calls in one block
            "<function_calls>[add(a=1,b=2), add(a=3,b=4)]</function_calls>",
            // Sentinel-shaped text that isn't a sentinel
            "the docs say <function_call but it's just text",
            // Call wrapped in surrounding text
            "prefix <function_calls>add(a=1,b=2)</function_calls> suffix",
            // String args with single quotes
            "<function_calls>[add(a=1, b=2)]</function_calls> trailing",
            // Empty trailing text
            "<function_calls>[add(a=1,b=2)]</function_calls>",
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
