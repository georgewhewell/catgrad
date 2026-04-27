//! Phi-3 / Phi-4-mini tool-call protocol.
//!
//! # Wire format
//!
//! Phi-4-mini emits tool calls as a single **prefix-only** sentinel
//! `functools` followed immediately by a JSON list of calls, with
//! **no closing sentinel** — the list ends at end-of-text:
//!
//! ```text
//! functools[{"name": "get_weather", "arguments": {"city": "Paris"}}]
//! ```
//!
//! Multiple calls live in the same JSON array (parallel function
//! calling):
//!
//! ```text
//! functools[{"name":"a","arguments":{...}},{"name":"b","arguments":{...}}]
//! ```
//!
//! ## Sources
//!
//! Microsoft documents the format on the Phi-4-mini model card / Phi
//! Cookbook:
//!
//! > "If you decide to call functions, you should prefix function calls
//! > with the `functools` marker (no closing marker required), and all
//! > function calls should be generated in a single JSON list formatted
//! > as `functools[{"name": ..., "arguments": ...}, ...]`."
//!
//! The chat template at
//! `https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/tokenizer_config.json`
//! does **not** itself render a tool-call branch (it only handles the
//! `<|tool|>...<|/tool|>` wrapping of tool *definitions* in the system
//! message); the `functools[...]` shape is what the model is fine-tuned
//! to emit during generation, mirrored by the canonical reference
//! parser in vLLM:
//!
//!   <https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/phi4mini_tool_parser.py>
//!
//! whose `bot_token = "functools"` and pattern `functools\[(.*?)\]`
//! confirm the prefix-only-sentinel shape.
//!
//! # State machine
//!
//! Unlike Qwen3 / LFM2 which use paired open/close sentinels, this
//! protocol has only an opening sentinel:
//!
//! - `Outside`: scan for `functools`. Anything before is `TextDelta`.
//! - `Inside`: buffer EVERYTHING after the prefix until `finish()`. There
//!   is no close sentinel; end-of-stream commits the parse.
//! - `Terminated`: a fatal error has been emitted; `feed`/`finish`
//!   return empty.
//!
//! # Strict gating
//!
//! Without the `functools` prefix, all input — even payloads that LOOK
//! like a JSON list of calls — is plain text. This matches the
//! qwen3 / lfm2 strict-gating rule: a Markdown-rendered example tool
//! call inside an explanation must not be parsed as a real call.
//!
//! # Multi-call atomicity
//!
//! The full JSON array is parsed before any call is emitted. If call
//! N+1 fails validation (unknown tool, schema-invalid args), calls
//! 0..N have already been emitted as full `Start` / `ArgsDelta` / `End`
//! triples and the parser then transitions to `Terminated`.

// All production wiring lives in `protocol.rs`'s `PHI4` registry
// entry. The dialect is a prefix sentinel (`functools`) followed by
// JSON list/object — no closing marker, payload drains at EOS.

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory,
};
#[cfg(test)]
use crate::runtime::chat::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;

#[cfg(test)]
fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    use crate::runtime::chat::tool_protocol_for;
    tool_protocol_for("Phi4ForCausalLM", &serde_json::Value::Null)
        .expect("Phi-4 arch is registered")
        .make_parser(directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;

    /// Universal sentinel-engine scenarios via the shared harness.
    /// Wire-format-specific cases live in the per-test sections below.
    #[test]
    fn passes_universal_scenarios() {
        use crate::runtime::chat::protocol_test_kit::{ProtocolTestFixture, directory_with_add};
        ProtocolTestFixture {
            make_parser: Box::new(make_parser),
            directory: directory_with_add(),
            valid_call_add_1_2: r##"functools[{"name":"add","arguments":{"a":1,"b":2}}]"##,
            unknown_tool_call: r##"functools[{"name":"missing","arguments":{}}]"##,
            invalid_args_call: r##"functools[{"name":"add","arguments":{"a":"x","b":2}}]"##,
            malformed_payload: r##"functoolsnot json"##,
            open_sentinel_only: Some(r##"functools"##),
        }
        .run_universal_scenarios();
    }

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


    #[test]
    fn valid_single_call_emits_on_finish() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"functools[{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::ToolCallArgsDelta { index: 0, .. }
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_multiple_calls_in_one_block() {
        let mut p = make_parser(directory_with_add_and_mul());
        let events = run(
            &mut *p,
            &[
                r#"functools[{"name":"add","arguments":{"a":1,"b":2}},{"name":"mul","arguments":{"a":3,"b":4}}]"#,
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
    fn text_before_sentinel_emits_as_text() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"sure, calling tool: functools[{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        // TextDelta("sure, calling tool: "), Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 5);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "sure, calling tool: "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }




    #[test]
    fn empty_payload_is_terminal() {
        // Just `functools` with no JSON before EOS.
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["functools"]);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: a payload that LOOKS like a tool call but lacks
        // the `functools` prefix must NOT be parsed.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
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
    fn partial_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble func"));
        // Nothing committed yet — `func` could extend into `functools`.
        // (`preamble ` is safe and may be emitted; that is fine.)
        events.extend(p.feed(r#"tools[{"name":"add","arguments":{"a":1,"b":2}}]"#));
        events.extend(p.finish(StopReason::EndOfText));
        // Combine the leading TextDeltas to verify they reassemble to "preamble ".
        let mut text = String::new();
        let mut idx = 0;
        while idx < events.len() {
            if let DecodeEvent::TextDelta(s) = &events[idx] {
                text.push_str(s);
                idx += 1;
            } else {
                break;
            }
        }
        assert_eq!(text, "preamble ");
        assert!(matches!(&events[idx], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }



    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // First call validates, second is unknown. The first triple
        // must be emitted, then UnknownTool + Stop{ProtocolError}.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                r#"functools[{"name":"add","arguments":{"a":1,"b":2}},{"name":"delete_db","arguments":{}}]"#,
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
    fn bare_object_payload_accepted() {
        // A single call without the surrounding `[]`.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"functools{"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameters_key_accepted_as_arguments() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"functools[{"name":"add","parameters":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

}

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across arbitrary boundaries produces the same
    //! final decoded turn (or the same DecodeFailure).

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // single valid call
            r#"functools[{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // call with text prefix
            r#"sure: functools[{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // bare-object payload
            r#"functools{"name":"add","arguments":{"a":1,"b":2}}"#,
            // two calls in one block
            r#"functools[{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]"#,
            // sentinel-shaped text that isn't the sentinel
            "the docs say functo... but it's just text",
            // unknown tool — chunk-invariance still holds (same fatal either way)
            r#"functools[{"name":"missing","arguments":{}}]"#,
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
