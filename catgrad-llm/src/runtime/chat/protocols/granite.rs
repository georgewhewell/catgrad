//! IBM Granite 3.x tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble text<|tool_call|>[{"name": "x", "arguments": {...}}, ...]
//! ```
//!
//! The Granite 3.3 chat template
//! (<https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/raw/main/tokenizer_config.json>)
//! instructs the model:
//!
//! > "When a tool is required to answer the user's query, respond only
//! > with `<|tool_call|>` followed by a JSON list of tools used."
//!
//! The added-tokens table (`added_tokens.json`) places `<|tool_call|>`
//! at id 49154 alongside other Granite control tokens.
//!
//! The format is **prefix-only**: there is no closing sentinel. Once the
//! model emits `<|tool_call|>`, every subsequent token belongs to the
//! tool-call payload, terminated by the natural end of the JSON array
//! plus EOS (`<|end_of_text|>`, id 0). Trailing whitespace after the
//! closing `]` is tolerated.
//!
//! The payload is a JSON list (possibly with a single element). Each
//! element carries `{"name": "...", "arguments": {...}}`; the alternate
//! `parameters` key is also accepted (matches the qwen3 / lfm2 dialects
//! seen in the wild). A bare object (no enclosing list) is also accepted
//! as a single call — the vLLM reference parser leaves this implicit but
//! some Granite checkpoints emit it.
//!
//! # Multiple calls per block
//!
//! Granite supports parallel calls inside the single JSON array. The
//! parser emits one `ToolCallStart` / `ToolCallArgsDelta` / `ToolCallEnd`
//! triple per validated call, with strictly increasing indices. If call
//! N+1 fails validation, calls 0..N are surfaced first and the fatal
//! event terminates the parser (matches the lfm2 partial-then-fatal
//! contract).
//!
//! # References
//!
//! - HF chat template: `tokenizer_config.json` (linked above).
//! - vLLM reference parser:
//!   <https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/granite_tool_parser.py>
//!   — strips `<|tool_call|>` (3.0) or `<tool_call>` (3.1) prefix, then
//!   parses the rest as a JSON list.

use std::sync::Arc;


use crate::runtime::chat::codecs::JsonObjectOrArrayCodec;
use crate::runtime::chat::sentinel_engine::{MAX_TOOL_CALL_PAYLOAD_BYTES, SentinelEngine};
use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory,
};

/// Granite 3.x prefix sentinel. Once seen, every subsequent byte belongs
/// to the tool-call payload (a JSON list of calls). There is no closing
/// sentinel — the payload terminates at EOS.
const TOOL_CALL_OPEN: &str = "<|tool_call|>";

/// Construct a Granite 3.x parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(GraniteParser::new(directory))
}

/// Render the bound tool list into the JSON shape the Granite 3.x chat
/// template expects.
///

struct GraniteParser(SentinelEngine);

impl GraniteParser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self(SentinelEngine::new_prefix(
            directory,
            Box::new(JsonObjectOrArrayCodec::strict()),
            TOOL_CALL_OPEN,
        ))
    }
}

impl IncrementalToolCallParser for GraniteParser {
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
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;

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

    fn mul_tool() -> ToolSpec {
        ToolSpec::new(
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
        )
    }

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
    }

    fn directory_with_add_and_mul() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool(), mul_tool()]).unwrap())
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

    /// All universal sentinel-engine behaviours go through the
    /// shared harness — this single test replaces what used to be
    /// 8+ separate per-protocol scenarios.
    #[test]
    fn passes_universal_scenarios() {
        use crate::runtime::chat::protocol_test_kit::ProtocolTestFixture;
        ProtocolTestFixture {
            make_parser: Box::new(make_parser),
            directory: directory_with_add(),
            valid_call_add_1_2: r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#,
            unknown_tool_call: r#"<|tool_call|>[{"name":"missing","arguments":{}}]"#,
            invalid_args_call: r#"<|tool_call|>[{"name":"add","arguments":{"a":"x","b":2}}]"#,
            malformed_payload: "<|tool_call|>not json at all",
            open_sentinel_only: Some("<|tool_call|>"),
        }
        .run_universal_scenarios();
    }


    #[test]
    fn valid_call_emits_start_args_end() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { index: 0, .. }));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn bare_object_payload_accepted_as_single_call() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>{"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameters_key_accepted_as_arguments() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"add","parameters":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn multiple_calls_in_block() {
        let mut p = GraniteParser::new(directory_with_add_and_mul());
        let events = run(
            &mut p,
            &[
                r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"mul","arguments":{"a":3,"b":4}}]"#,
            ],
        );
        // Two triples + Stop = 7 events.
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
    fn text_then_call_in_single_feed() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"first <|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "first "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }




    #[test]
    fn missing_name_field_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"arguments":{"a":1,"b":2}}]"#],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::MissingField("name")));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(&mut p, &["<|tool_call|>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_array_payload_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(&mut p, &["<|tool_call|>[]"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn partial_sentinel_split_across_feeds() {
        let mut p = GraniteParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <|tool_c"));
        // Held-back partial must not have leaked an opening `<` past
        // the safe-emit boundary.
        for e in &events {
            if let DecodeEvent::TextDelta(s) = e {
                assert!(!s.contains("<|tool_c"), "leaked partial sentinel: {s:?}");
            }
        }
        events.extend(p.feed(r#"all|>[{"name":"add","arguments":{"a":1,"b":2}}]"#));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "preamble "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn payload_split_across_feeds_assembles_at_finish() {
        let mut p = GraniteParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(r#"<|tool_call|>[{"name":"add","argum"#));
        // No close sentinel exists — no events should have been emitted
        // mid-payload.
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed(r#"ents":{"a":1,"b":2}}]"#));
        // Still nothing — payload only completes at finish().
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_payload_without_sentinel_is_plain_text() {
        // Bare JSON resembling a tool call must NOT be parsed.
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"Here is some JSON: [{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == r#"Here is some JSON: [{"name":"add","arguments":{"a":1,"b":2}}]"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unterminated_block_at_eos_with_full_json_is_ok() {
        // Granite has no closing sentinel — a complete JSON list is a
        // valid call even if the stream ends right after `]`.
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unterminated_block_at_eos_with_partial_json_is_terminal() {
        // Partial JSON at EOS (no `]` yet) is a parse error.
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"add","arguments":{"a":1"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }



    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // First call validates, second is unknown — the first triple
        // must be surfaced before UnknownTool/Stop.
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"delete_db","arguments":{}}]"#,
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
    fn sentinel_prefix_that_doesnt_resolve_emits_as_text() {
        let mut p = GraniteParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("hello <|to"));
        events.extend(p.feed("morrow"));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "hello <|tomorrow");
    }


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
            // single valid call (JSON list of one)
            r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // bare object form
            r#"<|tool_call|>{"name":"add","arguments":{"a":1,"b":2}}"#,
            // call with text prefix only (no postamble — Granite has
            // no close sentinel so anything after the payload would be
            // appended to it)
            r#"prefix <|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // two calls in one block
            r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]"#,
            // sentinel-shaped text that isn't a sentinel
            "the docs say <|tool_call but it's just text",
            // unknown tool (chunk-invariance still holds — same failure
            // either way)
            r#"<|tool_call|>[{"name":"missing","arguments":{}}]"#,
            // payload with whitespace around the JSON
            "<|tool_call|> \n[{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}]\n",
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
