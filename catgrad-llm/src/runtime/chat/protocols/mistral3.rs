//! Mistral / Ministral 3 tool-call protocol.
//!
//! # Wire format
//!
//! Mistral V3 tokenizer-family models (Mistral-7B-Instruct-v0.3,
//! Ministral-8B-Instruct-2410, Mistral-3 / Ministral-3) emit tool calls
//! using a single **prefix sentinel `[TOOL_CALLS]`** followed by a JSON
//! payload, then EOS. There is **no closing sentinel** — once
//! `[TOOL_CALLS]` is observed, the rest of the stream up to EOS is the
//! payload, parsed atomically at `finish()`.
//!
//! Canonical example (verbatim from the Mistral-7B-Instruct-v0.3
//! `chat_template` `tool_calls` branch):
//!
//! ```text
//! prefix [TOOL_CALLS] [{"name": "calculator", "arguments": {"a": 1, "b": 2}, "id": "abc123xyz"}, {"name": "x", "arguments": {}, "id": "def456uvw"}]</s>
//! ```
//!
//! Confirmed via:
//! - Mistral-7B-Instruct-v0.3 tokenizer config — token id 5 is
//!   `[TOOL_CALLS]`; the chat template renders
//!   `[TOOL_CALLS] [{...function-tojson..., "id": "..."}, ...]` and
//!   then `eos_token`. There is no `[/TOOL_CALLS]` token.
//! - Ministral-8B-Instruct-2410 tokenizer config — same `[TOOL_CALLS]`
//!   token; template renders `[TOOL_CALLS][...]` (no leading space).
//! - vLLM `mistral_tool_parser.py` — bot_token = `"[TOOL_CALLS] ["`,
//!   payload is `[{"name": "...", "arguments": {...}, ...}, ...]`,
//!   args field is `arguments`.
//! - SGLang `mistral_detector.py` — searches for `[TOOL_CALLS`,
//!   payload is a JSON array of objects with `name` + `arguments`.
//!
//! # Field names
//!
//! Each call object is `{"name": <str>, "arguments": <object>, "id":
//! <9-char-string>}`. The `id` is template-required for the model's own
//! tool-result threading but is irrelevant to the parser — we read
//! `name` and `arguments`. We also accept `parameters` as an alias for
//! `arguments` to match the rest of the codebase's permissiveness
//! (Qwen3 / LFM2 do the same), even though Mistral's official template
//! always emits `arguments`.
//!
//! # Payload shape
//!
//! The official template always emits a JSON **list** wrapping the call
//! objects (even for a single call). Real-world model output sometimes
//! drops the outer brackets and emits a bare object; vLLM accepts that,
//! so we do too.
//!
//! # No trailing prose
//!
//! Once `[TOOL_CALLS]` opens, no return to user-visible text is
//! permitted before EOS. This matches the chat-template structure (the
//! `tool_calls` branch ends with `eos_token` immediately after the JSON
//! list) and matches both vLLM and SGLang. Anything after the JSON list
//! parse cleanly consumes — anything after that on the wire would be
//! garbage we cannot interpret as text retroactively. We do not emit a
//! tail `TextDelta`.
//!
//! # Strict gating
//!
//! Bare JSON without the `[TOOL_CALLS]` sentinel is plain text. This
//! matches the strict-gating posture of the Qwen3 / LFM2 parsers — a
//! model output that happens to contain `{"name": "x", "arguments":
//! {...}}` is not a tool call unless preceded by `[TOOL_CALLS]`.
//!
//! # Render shape
//!
//! `render_tools` produces the OpenAI envelope
//! `[{"type": "function", "function": {name, description, parameters}}, ...]`.
//! The Mistral-7B-Instruct-v0.3 and Ministral chat templates iterate
//! `tools` and unpack `tool.function`, so this shape is what the
//! template expects.

// All production wiring lives in `protocol.rs`'s `MISTRAL3` registry
// entry. The dialect is a prefix sentinel (`[TOOL_CALLS]`) followed by
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
    tool_protocol_for("MistralForCausalLM", &serde_json::Value::Null)
        .expect("Mistral arch is registered")
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



    #[test]
    fn valid_single_call_emits_start_args_end_on_finish() {
        let mut p = make_parser(directory_with_add());
        // Feed only — no Start/Args/End events should be emitted yet:
        // the payload is buffered until finish() because there is no
        // closing sentinel.
        let mid_events =
            p.feed(r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}}]"#);
        assert!(
            !mid_events.iter().any(|e| matches!(
                e,
                DecodeEvent::ToolCallStart { .. }
                    | DecodeEvent::ToolCallArgsDelta { .. }
                    | DecodeEvent::ToolCallEnd { .. }
            )),
            "got events during feed: {mid_events:?}"
        );
        let final_events = p.finish(StopReason::EndOfText);
        // Start, ArgsDelta, End, Stop = 4
        assert_eq!(final_events.len(), 4);
        assert!(matches!(
            &final_events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(
            &final_events[1],
            DecodeEvent::ToolCallArgsDelta { index: 0, .. }
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &final_events[2] else {
            panic!("expected ToolCallEnd, got {:?}", final_events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert!(matches!(
            &final_events[3],
            DecodeEvent::Stop {
                reason: StopReason::EndOfText
            }
        ));
    }

    #[test]
    fn valid_multiple_calls_in_one_block() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]"#,
            ],
        );
        // Two triples + Stop = 7 events
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
    fn text_before_sentinel_emits_as_text() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"prefix [TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "prefix "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn bare_object_payload() {
        // Real model output sometimes drops the outer brackets; vLLM
        // accepts that and so do we.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"[TOOL_CALLS] {"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn no_space_after_sentinel_is_accepted() {
        // Ministral's chat template emits `[TOOL_CALLS][...]` with no
        // space between the sentinel and the bracket.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"[TOOL_CALLS][{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn id_field_is_ignored() {
        // The official template emits `{"name": ..., "arguments": ...,
        // "id": "9-char"}` — extra fields don't disturb us.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2},"id":"abc123xyz"}]"#,
            ],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameters_alias_is_accepted() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"[TOOL_CALLS] [{"name":"add","parameters":{"a":1,"b":2}}]"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn arguments_as_json_encoded_string_is_accepted() {
        // OpenAI-legacy shape: `arguments` is a JSON-encoded string.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"[TOOL_CALLS] [{"name":"add","arguments":"{\"a\":1,\"b\":2}"}]"#],
        );
        assert_eq!(events.len(), 4);
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }




    #[test]
    fn empty_payload_is_terminal() {
        // Just `[TOOL_CALLS]` with nothing after.
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["[TOOL_CALLS]"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_array_payload_is_terminal() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["[TOOL_CALLS] []"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: bare JSON resembling a tool call must NOT be parsed
        // without the `[TOOL_CALLS]` sentinel.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"Here is some JSON: [{"name":"add","arguments":{"a":1,"b":2}}]"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s)
                if s == r#"Here is some JSON: [{"name":"add","arguments":{"a":1,"b":2}}]"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        // Split inside the sentinel.
        events.extend(p.feed("preamble [TOOL_C"));
        // Whatever is in `events` so far cannot include the `[T...` tail
        // — the matcher holds the partial-sentinel suffix back.
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                assert!(!s.contains('['), "unexpected `[` in TextDelta: {s:?}");
            }
        }
        events.extend(p.feed(r#"ALLS] [{"name":"add","arguments":{"a":1,"b":2}}]"#));
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
    fn payload_split_across_feeds_inside_buffer() {
        // Once Inside, we keep buffering across feeds until finish.
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(r#"[TOOL_CALLS] [{"name":"add",""#));
        events.extend(p.feed(r#"arguments":{"a":1,"#));
        events.extend(p.feed(r#""b":2}}]"#));
        // Nothing emitted yet — payload-parse only happens on finish.
        assert!(events.is_empty(), "got events during feed: {events:?}");
        events.extend(p.finish(StopReason::EndOfText));
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }


    #[test]
    fn after_fatal_via_oversize_subsequent_feed_returns_empty() {
        // Oversize triggers fatal during feed (eagerly), which gives us
        // a different path through the state machine to test that
        // post-fatal feed/finish still return empty.
        let mut p = make_parser(directory_with_add());
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("[TOOL_CALLS]"));
        events.extend(p.feed(&oversize));
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}")
        };
        assert!(matches!(
            source,
            ParserError::PayloadTooLarge { limit_bytes }
                if *limit_bytes == MAX_TOOL_CALL_PAYLOAD_BYTES
        ));
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }


    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // First call is valid `add`, second is unknown — must emit the
        // first call's full triple before the UnknownTool fatal.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[
                r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}},{"name":"delete_db","arguments":{}}]"#,
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
    fn missing_name_field_is_terminal() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &[r#"[TOOL_CALLS] [{"arguments":{}}]"#]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn arguments_not_object_is_terminal() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"[TOOL_CALLS] [{"name":"add","arguments":42}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
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
            // single valid call (with leading space — Mistral 7B form)
            r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // single valid call (no space — Ministral form)
            r#"[TOOL_CALLS][{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // bare object
            r#"[TOOL_CALLS] {"name":"add","arguments":{"a":1,"b":2}}"#,
            // call with `id` field (real Mistral output)
            r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2},"id":"abc123xyz"}]"#,
            // two calls in one block
            r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]"#,
            // call with prefix text
            r#"prefix text [TOOL_CALLS] [{"name":"add","arguments":{"a":1,"b":2}}]"#,
            // sentinel-shaped text that isn't a sentinel
            "the docs say [TOOL but it's just text",
            // unknown tool — produces a fatal event; chunk-invariance
            // still holds.
            r#"[TOOL_CALLS] [{"name":"missing","arguments":{}}]"#,
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..9,
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
            input_idx in 0_usize..9,
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
