//! NVIDIA Nemotron / Nemotron-H tool-call protocol.
//!
//! # Wire format research
//!
//! The reference chat template
//! (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/chat_template.jinja`) renders
//! tool calls as a Llama-3-style XML payload, *not* the Hermes-style
//! JSON of Qwen3:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY1>
//! VALUE1
//! </parameter>
//! <parameter=KEY2>
//! VALUE2
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Each `<tool_call>...</tool_call>` block carries exactly one
//! `<function=...>...</function>` invocation; multiple parallel calls
//! arrive as multiple back-to-back `<tool_call>` blocks (the chat
//! template loops over `message.tool_calls` and emits one block per
//! call).
//!
//! Per-parameter values are stringified via Jinja's `string` filter for
//! scalars and `tojson` for mappings/sequences. To round-trip, this
//! parser tries `serde_json::from_str` on the trimmed value first and
//! falls back to a JSON string on failure — matching the historical
//! `parse_qwen3_5_tool_calls` reference in
//! `catgrad-llm/src/helpers/tool_calls.rs` (now removed; see git
//! history at `675fd07~1`).
//!
//! # Why standalone, not a Qwen3 delegate
//!
//! The outer sentinels `<tool_call>` / `</tool_call>` happen to be
//! identical to Qwen3 — but the payload between them is XML, not
//! Hermes JSON. Delegating to `qwen3::make_parser` would treat the XML
//! body as malformed JSON and fatal-out every call. This module
//! therefore reimplements the state machine (Outside / Inside /
//! Terminated) with an XML payload parser inside the Inside branch.
//!
//! The protocol-registry comment in
//! `runtime/chat/protocol.rs:120` ("Hermes-style JSON in
//! `<tool_call>`") refers to an earlier expectation of the architecture
//! and is inaccurate for the production Nemotron-3-Nano chat template.
//! The registry only stores function pointers, so the wrong comment is
//! cosmetic.
//!
//! # Per-call atomic emission
//!
//! As with the Qwen3 parser, `<tool_call>` opens a buffering mode and
//! the `Start` / `ArgsDelta` / `End` triple is emitted atomically when
//! the matching `</tool_call>` arrives. True per-token argument
//! streaming is out of scope.
//!
//! # Strict gating
//!
//! Bare `<function=...>` syntax outside a `<tool_call>` wrapper is
//! plain text, not a tool call — same rule as the Qwen3 parser's
//! treatment of bare JSON.

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
    tool_protocol_for("NemotronForCausalLM", &serde_json::Value::Null)
        .expect("arch is registered")
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
            valid_call_add_1_2: r##"<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>"##,
            unknown_tool_call: r##"<tool_call><function=missing></function></tool_call>"##,
            invalid_args_call: r##"<tool_call><function=add><parameter=a>"x"</parameter><parameter=b>2</parameter></function></tool_call>"##,
            malformed_payload: r##"<tool_call>not xml</tool_call>"##,
            open_sentinel_only: Some(r##"<tool_call>"##),
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

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
    }

    /// Render a Nemotron tool-call block. Mirrors the chat template
    /// shape, including the synthetic newlines around values.
    fn call_block(name: &str, args: &[(&str, &str)]) -> String {
        let mut out = String::from("<tool_call>\n<function=");
        out.push_str(name);
        out.push_str(">\n");
        for (k, v) in args {
            out.push_str("<parameter=");
            out.push_str(k);
            out.push_str(">\n");
            out.push_str(v);
            out.push_str("\n</parameter>\n");
        }
        out.push_str("</function>\n</tool_call>");
        out
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
    fn valid_call_emits_start_args_end() {
        let mut p = make_parser(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let events = run(&mut *p, &[&block]);
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { index: 0, .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn multiple_calls_in_sequence() {
        let mul = ToolSpec::new(
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
        );
        let dir = Arc::new(ToolDirectory::new(vec![add_tool(), mul]).unwrap());
        let mut p = make_parser(dir);
        let first = call_block("add", &[("a", "1"), ("b", "2")]);
        let second = call_block("mul", &[("a", "3"), ("b", "4")]);
        let events = run(&mut *p, &[&first, &second]);
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
    fn text_then_call_then_text_in_single_feed() {
        let mut p = make_parser(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let combined = format!("first {block} last");
        let events = run(&mut *p, &[&combined]);
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
    fn missing_name_field_is_terminal_with_protocol_error() {
        // `<function=>` — empty name.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["<tool_call><function=></function></tool_call>"],
        );
        assert_eq!(events.len(), 2);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::MissingField("name")));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn parameters_key_is_accepted_as_arguments() {
        // The Nemotron wire format does not use a JSON envelope with
        // `arguments` / `parameters` keys (each parameter is its own
        // XML block). The test name is part of the contract checklist;
        // for this protocol the analogous concern is that
        // `<parameter=KEY>` *is* the way to carry arguments — and that
        // we read both numeric and unquoted-string scalar forms back
        // into a plain JSON object.
        let mut p = make_parser(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let events = run(&mut *p, &[&block]);
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {events:?}");
        };
        // Arguments arrive as a plain JSON object keyed by parameter name.
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: a JSON-shaped payload not wrapped in `<tool_call>`
        // is plain text — same strict-gating rule as Qwen3.
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        // No call can have committed yet; no partial-sentinel bytes
        // should leak as text.
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains('<'),
            _ => true,
        }));
        events.extend(p.feed(
            "all>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> done",
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
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_",
        ));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["<tool_call>\n<function=add>\n<parameter=a>\n1"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }




    // Codec helper tests (parse_scalar / parse_function_block)
    // moved into `codecs::xml_function`.
}

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across chunk boundaries produces the same final
    //! decoded turn (or the same DecodeFailure).

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // single valid call (matching the `add` tool used by the
            // shared test directory in `test_util`)
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>",
            // call surrounded by text
            "prefix <tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> suffix",
            // two calls back-to-back
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=add>\n<parameter=a>\n3\n</parameter>\n<parameter=b>\n4\n</parameter>\n</function>\n</tool_call>",
            // sentinel-shaped text that isn't a sentinel
            "the docs say <tool_call> but it's just text",
            // unknown tool — fatal, but chunk-invariance still holds
            "<tool_call>\n<function=missing>\n</function>\n</tool_call>",
            // call with text suffix only
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> done",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..7,
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
            input_idx in 0_usize..7,
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
