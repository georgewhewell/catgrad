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
