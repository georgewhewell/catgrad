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
    use crate::runtime::chat::protocol_test_kit::{
        add_tool, directory_with_add, last_stop_reason, run,
    };
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;


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

    fn directory_with_add_and_mul() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool(), mul_tool()]).unwrap())
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



}
