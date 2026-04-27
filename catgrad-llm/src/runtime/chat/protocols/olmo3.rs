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
