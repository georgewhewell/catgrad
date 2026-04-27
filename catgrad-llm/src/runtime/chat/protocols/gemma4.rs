//! Gemma 4 tool-call protocol.
//!
//! Wire format (post-detokenization, with `skip_special_tokens=false`):
//!
//! ```text
//! preamble text<|tool_call>call:NAME{key1:value1,key2:value2,...}<tool_call|>
//! ```
//!
//! The Gemma 4 chat template emits each call as a single `<|tool_call>`
//! ... `<tool_call|>` block. Inside the block, the body has the shape
//! `call:NAME{...}` where `{...}` is a comma-separated list of
//! `key:value` pairs (keys are bare identifiers, values use the
//! template's `format_argument` shape — see below).
//!
//! # Value encoding
//!
//! `format_argument` (`escape_keys=False` at the top level and
//! recursively) renders argument values as:
//!
//! - String:  `<|"|>VALUE<|"|>` — VALUE is literal, no escaping.
//! - Boolean: `true` / `false`.
//! - Mapping: `{key:value,key:value,...}` — keys remain bare.
//! - Array:   `[value,value,...]`.
//! - Other:   raw textual representation (numbers, null/None).
//!
//! Note `<|"|>` is the same opening and closing sentinel — strings are
//! delimited by paired occurrences of the same token. There is no
//! escape mechanism, so the contract here is "string values cannot
//! literally contain `<|"|>`".
//!
//! # Streaming guarantee
//!
//! `<|tool_call>` opens a buffering mode; only when `<tool_call|>`
//! arrives do we parse, validate, and emit the
//! `ToolCallStart` + `ToolCallArgsDelta` + `ToolCallEnd` triple as one
//! atomic unit.
//!
//! # Decoding requirement
//!
//! All three sentinels (`<|tool_call>`, `<tool_call|>`, `<|"|>`) are
//! marked `special: true` in the tokenizer. Callers MUST detokenize
//! with `skip_special_tokens=false` for parser input — otherwise the
//! sentinels are stripped before this code can see them and tool calls
//! become invisible. The chat-aware example/server paths already do
//! this; the parser asserts nothing about it because by the time text
//! reaches `feed()` it is already a string.

// All production wiring lives in `protocol.rs`'s `GEMMA4` registry
// entry. Wire format: asymmetric sentinel pair (`<|tool_call>` /
// `<tool_call|>`) wrapping `call:NAME{key:value, ...}` with
// `<|"|>`-quoted strings. Parsed by `codecs::Gemma4Codec`.

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::runtime::chat::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;
#[cfg(test)]
use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory,
};
#[cfg(test)]
use serde_json::Value as JsonValue;

#[cfg(test)]
fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    use crate::runtime::chat::tool_protocol_for;
    tool_protocol_for("Gemma4ForConditionalGeneration", &serde_json::Value::Null)
        .expect("Gemma 4 arch is registered")
        .make_parser(directory)
}

/// Local helper kept in the test module since two scenario tests
/// assert on the parsed-args defensive check that the codec performs.
#[cfg(test)]
fn args_contain_literal_quote_sentinel(value: &JsonValue) -> bool {
    match value {
        JsonValue::String(s) => s.contains("<|\"|>"),
        JsonValue::Array(items) => items.iter().any(args_contain_literal_quote_sentinel),
        JsonValue::Object(map) => map.values().any(args_contain_literal_quote_sentinel),
        _ => false,
    }
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
            valid_call_add_1_2: r##"<|tool_call>call:add{a:1,b:2}<tool_call|>"##,
            unknown_tool_call: r##"<|tool_call>call:missing{}<tool_call|>"##,
            invalid_args_call: r##"<|tool_call>call:add{a:<|"|>x<|"|>,b:2}<tool_call|>"##,
            malformed_payload: r##"<|tool_call>this isn't valid<tool_call|>"##,
            open_sentinel_only: Some(r##"<|tool_call>"##),
        }
        .run_universal_scenarios();
    }

    fn calculator_tool() -> ToolSpec {
        ToolSpec::new(
            "calculator",
            Some("simple calculator".into()),
            json!({
                "type": "object",
                "properties": {
                    "lhs": { "type": "number" },
                    "rhs": { "type": "number" },
                    "op": { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                },
                "required": ["lhs", "rhs", "op"],
            }),
        )
    }

    fn directory() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![calculator_tool()]).unwrap())
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
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:calculator{lhs:1353785,op:<|"|>div<|"|>,rhs:790489}<tool_call|>"#],
        );
        let mut iter = events.iter();
        let first = iter.next().expect("Start event");
        match first {
            DecodeEvent::ToolCallStart { index: 0, name } => assert_eq!(name, "calculator"),
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        let args_delta = iter.next().expect("ArgsDelta event");
        match args_delta {
            DecodeEvent::ToolCallArgsDelta { index: 0, delta } => {
                let parsed: JsonValue = serde_json::from_str(delta).unwrap();
                assert_eq!(parsed["lhs"], 1353785);
                assert_eq!(parsed["rhs"], 790489);
                assert_eq!(parsed["op"], "div");
            }
            other => panic!("expected ToolCallArgsDelta, got {other:?}"),
        }
        let end = iter.next().expect("End event");
        match end {
            DecodeEvent::ToolCallEnd { index: 0, args } => {
                assert_eq!(args["op"], "div");
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn call_emitted_atomically_when_close_sentinel_arrives() {
        let mut p = make_parser(directory());
        let events = p.feed(
            r#"<|tool_call>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|>"#,
        );
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::Stop { .. })));
    }



    #[test]
    fn missing_call_prefix_is_terminal() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>calculator{lhs:1}<tool_call|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_block_is_terminal() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:calculator{lhs:1"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::Unterminated,
                ..
            }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = make_parser(directory());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <|tool_c"));
        events.extend(p.feed(
            r#"all>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|> done"#,
        ));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "preamble  done");
        assert!(events.iter().any(|e| matches!(e, DecodeEvent::ToolCallStart { .. })));
    }

    #[test]
    fn partial_string_quote_split_across_feeds() {
        let mut p = make_parser(directory());
        let mut events = Vec::new();
        events.extend(p.feed(r#"<|tool_call>call:calculator{lhs:1,op:<|"#));
        events.extend(p.feed(r#""|>add<|"|>,rhs:2}<tool_call|>"#));
        events.extend(p.finish(StopReason::EndOfText));
        // The args should still parse correctly as op=add despite split
        // mid-quote sentinel.
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["op"], "add");
    }

    #[test]
    fn multiple_calls_in_sequence() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[
                r#"<|tool_call>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|>"#,
                r#"<|tool_call>call:calculator{lhs:3,op:<|"|>mul<|"|>,rhs:4}<tool_call|>"#,
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "calculator"
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, name } if name == "calculator"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn empty_args_object_parses() {
        // A tool with no required args could legitimately emit `{}`.
        let nullary_tool = ToolSpec::new(
            "ping",
            None,
            json!({ "type": "object", "properties": {} }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![nullary_tool]).unwrap());
        let mut p = make_parser(dir);
        let events = run(&mut *p, &[r#"<|tool_call>call:ping{}<tool_call|>"#]);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_text_resembling_call_without_sentinel_is_plain_text() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"call:calculator{lhs:1,op:add,rhs:2}"#],
        );
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, r#"call:calculator{lhs:1,op:add,rhs:2}"#);
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }


    // -- Real-world failure modes (sourced from the upstream parsers
    //    in vLLM, SGLang, and llama.cpp). See the inline comments for
    //    the originating issue.

    fn search_tool() -> ToolSpec {
        ToolSpec::new(
            "tools.shell-exec",
            None,
            json!({
                "type": "object",
                "properties": {
                    "cmd": { "type": "string" },
                    "args": { "type": "array", "items": { "type": "string" } },
                    "env": { "type": "object" },
                    "limit": { "type": "integer" },
                    "dry_run": { "type": "boolean" },
                },
                "required": ["cmd"],
            }),
        )
    }

    fn search_directory() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![search_tool()]).unwrap())
    }

    /// Function names with `-` and `.` are common in real tool catalogs
    /// (e.g. `tools.shell-exec`). vLLM accepts `[\w\-\.]+`; we match.
    #[test]
    fn function_name_with_dot_and_dash_accepted() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>ls<|"|>}<tool_call|>"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "tools.shell-exec"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    /// Function names that don't fit the documented charset (whitespace,
    /// leading digit, etc.) are rejected loudly as protocol errors —
    /// silently passing them through risks the executor blowing up on a
    /// `tool_call` whose `function.name` it cannot dispatch.
    #[test]
    fn invalid_function_name_charset_is_rejected() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:bad name{a:1}<tool_call|>"#],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(
            source.to_string().contains("invalid function name"),
            "got: {source}"
        );
    }

    /// llama.cpp #21384 / #21316: braces inside a string value broke the
    /// outer object's brace matcher. The fix is to skip over `<|"|>...
    /// <|"|>` regions during depth counting — this test pins it.
    #[test]
    fn string_value_with_braces_is_opaque() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>echo {hello, world}<|"|>}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["cmd"], "echo {hello, world}");
    }

    /// Same defense as above for arrays inside strings — `[`/`]`
    /// inside a `<|"|>...<|"|>` region must not flip array depth.
    #[test]
    fn string_value_with_brackets_is_opaque() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>grep [abc] /etc/hosts<|"|>}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["cmd"], "grep [abc] /etc/hosts");
    }

    /// Nested object arguments parse recursively. Keys remain bare at
    /// every level (`escape_keys=False` propagates).
    #[test]
    fn nested_object_argument() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>ls<|"|>,env:{HOME:<|"|>/root<|"|>,LANG:<|"|>C<|"|>}}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["env"]["HOME"], "/root");
        assert_eq!(args["env"]["LANG"], "C");
    }

    /// Array of strings — each element delimited by `<|"|>` and
    /// separated at the top level of the array by `,`.
    #[test]
    fn array_of_strings_argument() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>cargo<|"|>,args:[<|"|>build<|"|>,<|"|>--release<|"|>]}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["args"][0], "build");
        assert_eq!(args["args"][1], "--release");
    }

    /// Booleans arrive as bare `true` / `false`.
    #[test]
    fn boolean_argument() {
        let mut p = make_parser(search_directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:tools.shell-exec{cmd:<|"|>ls<|"|>,dry_run:true}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["dry_run"], true);
    }

    /// Negative integers and floats round-trip through `parse_number`.
    #[test]
    fn negative_and_float_numbers() {
        let nums = ToolSpec::new(
            "calc",
            None,
            json!({
                "type": "object",
                "properties": {
                    "i": { "type": "integer" },
                    "f": { "type": "number" },
                },
                "required": ["i", "f"],
            }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![nums]).unwrap());
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:calc{f:-3.14,i:-42}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["i"], -42);
        let f = args["f"].as_f64().unwrap();
        assert!((f - -3.14_f64).abs() < 1e-9, "got {f}");
    }

    /// HF discussions #20 / #55 on `google/gemma-4-*-it`: an outdated
    /// chat-template revision emitted `<|tool_call>{{...}}<tool_call|>`
    /// — JSON-shaped, not bare-key-form. Reject loudly with a hint
    /// pointing to the upstream issue rather than silently misparsing.
    #[test]
    fn outdated_double_braced_template_is_rejected_with_hint() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>{{"name":"calculator","arguments":{"lhs":1,"op":"add","rhs":2}}}<tool_call|>"#],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        let msg = source.to_string();
        assert!(
            msg.contains("double-braced") || msg.contains("outdated"),
            "expected hint about outdated template, got: {msg}"
        );
    }

    /// Trailing whitespace between `}` and `<tool_call|>` is tolerated.
    /// Some chat-template revisions add a stray newline before the
    /// close sentinel.
    #[test]
    fn trailing_whitespace_in_body_tolerated() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[
                "<|tool_call>call:calculator{lhs:1,op:<|\"|>add<|\"|>,rhs:2}\n<tool_call|>",
            ],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    /// Args arrive in any order — the chat template uses `dictsort` so
    /// alphabetical is the canonical wire form, but the parser itself
    /// must not depend on ordering.
    #[test]
    fn args_in_non_alphabetical_order_parse() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"<|tool_call>call:calculator{rhs:2,lhs:1,op:<|"|>add<|"|>}<tool_call|>"#],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = events
            .iter()
            .find(|e| matches!(e, DecodeEvent::ToolCallEnd { .. }))
            .expect("expected ToolCallEnd")
        else {
            unreachable!();
        };
        assert_eq!(args["lhs"], 1);
        assert_eq!(args["rhs"], 2);
    }

    /// Two parsers from the same directory are independent — vLLM
    /// shipped a bug (#39392) where tool-parser instance state was
    /// shared across requests, causing `<pad>` spam.  We construct
    /// per-request via `make_parser`, but the unit-level invariant is
    /// worth pinning: state mutations on parser A must not show up on
    /// parser B.
    #[test]
    fn parsers_have_independent_state() {
        let dir = directory();
        let mut a = make_parser(dir.clone());
        let mut b = make_parser(dir);
        a.feed(r#"<|tool_call>call:nope{}<tool_call|>"#); // poisons a
        // b must still accept a valid call cleanly.
        let events_b = run(
            &mut *b,
            &[r#"<|tool_call>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|>"#],
        );
        assert!(matches!(&events_b[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events_b), StopReason::EndOfText);
    }

    /// Defensive: `args_contain_literal_quote_sentinel` directly. The
    /// helper walks the parsed JSON looking for any string that
    /// retained a literal `<|"|>` — a smoke signal for upstream
    /// malformed sentinel pairing.
    #[test]
    fn args_contain_literal_quote_sentinel_helper() {
        assert!(!args_contain_literal_quote_sentinel(&json!({ "ok": "hi" })));
        assert!(args_contain_literal_quote_sentinel(
            &json!({ "bad": "leak <|\"|> here" })
        ));
        assert!(args_contain_literal_quote_sentinel(
            &json!({ "nested": [{ "x": "<|\"|>" }] })
        ));
    }

    /// Two valid calls separated by interleaved text. Indices are
    /// 0 and 1 (per `parser_index`); text is `TextDelta`.
    #[test]
    fn parallel_calls_with_interleaved_text() {
        let mut p = make_parser(directory());
        let events = run(
            &mut *p,
            &[r#"first <|tool_call>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|> middle <|tool_call>call:calculator{lhs:3,op:<|"|>mul<|"|>,rhs:4}<tool_call|> last"#],
        );
        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DecodeEvent::ToolCallStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![0, 1]);
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "first  middle  last");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            "hello world",
            r#"<|tool_call>call:add{a:1,b:2}<tool_call|>"#,
            r#"prefix <|tool_call>call:add{a:1,b:2}<tool_call|> suffix"#,
            r#"<|tool_call>call:add{a:1,b:2}<tool_call|><|tool_call>call:add{a:3,b:4}<tool_call|>"#,
            "the docs say <|tool_call> but it's just text",
            r#"<|tool_call>call:missing{}<tool_call|>"#,
            r#"<|tool_call>call:add{a:1,b:2}<tool_call|> done"#,
            // Strings with internal braces / brackets — must remain
            // opaque under any chunk boundary.
            r#"<|tool_call>call:add{a:<|"|>{not real}<|"|>,b:2}<tool_call|>"#,
            // Outdated chat-template payload (HF discussions #20/#55):
            // chunk-invariance still holds — same DecodeFailure either way.
            r#"<|tool_call>{{"name":"add","arguments":{"a":1,"b":2}}}<tool_call|>"#,
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
