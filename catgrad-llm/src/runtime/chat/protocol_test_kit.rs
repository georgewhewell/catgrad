//! Universal scenario harness for sentinel-engine protocols.
//!
//! Per-protocol tests historically duplicated ~10 universal scenarios
//! (plain-text passthrough, valid-call triple emission, unknown-tool /
//! invalid-args / malformed-payload terminal errors, after-fatal
//! poisoning, chunk-split invariance, UTF-8 safety, payload-size cap).
//! This module factors those into one [`ProtocolTestFixture`] +
//! [`ProtocolTestFixture::run_universal_scenarios`] pair.
//!
//! Per-protocol responsibilities shrink to **one fixture** + **one
//! test fn** that calls the harness:
//!
//! ```ignore
//! #[test]
//! fn granite_passes_universal_scenarios() {
//!     fixture().run_universal_scenarios();
//! }
//! ```
//!
//! Wire-format-specific tests stay in the per-protocol module —
//! anything that tests the dialect's quirks (parameters-key alias,
//! bare-object form, asymmetric-sentinel oddities) doesn't belong
//! here.

#![cfg(test)]

use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::event::{DecodeEvent, ParserError, StopReason};
use super::parser::IncrementalToolCallParser;
use super::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;
use super::tool_spec::ToolDirectory;

/// Constructor: `fn(directory) -> Box<dyn IncrementalToolCallParser>`.
/// Each protocol module supplies a closure that goes through the
/// registry, so the harness exercises the same parser path the
/// gateway uses.
pub type ParserFactory = Box<dyn Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>>;

/// Per-protocol inputs for the universal scenarios.
///
/// Each `&'static str` is the wire-format-encoded input for one
/// scenario. The harness asserts the resulting `DecodeEvent` shape;
/// the protocol tells the harness how to encode each scenario for
/// its dialect.
pub struct ProtocolTestFixture {
    /// Constructs a fresh parser bound to `directory`.
    pub make_parser: ParserFactory,
    /// Tool directory the fixture inputs are encoded against.
    /// Conventionally must contain a tool named `add` with required
    /// number args `a` and `b` (matched by `valid_single_call`).
    pub directory: Arc<ToolDirectory>,
    /// Wire-encoded valid call to `add` with `{a:1, b:2}`.
    pub valid_call_add_1_2: &'static str,
    /// Wire-encoded call to a tool name NOT in `directory`.
    pub unknown_tool_call: &'static str,
    /// Wire-encoded call to `add` with args that violate the schema
    /// (e.g. `a` is a string instead of a number).
    pub invalid_args_call: &'static str,
    /// Sentinel-opened payload that is malformed (e.g. invalid JSON
    /// between sentinels, unparseable Pythonic, etc.).
    pub malformed_payload: &'static str,
    /// Just the open sentinel string — used by the
    /// payload-too-large test to start a block then stuff an
    /// oversize body. Set to `None` for protocols that don't have
    /// a meaningful "inside-block byte buffer" (none today; all
    /// engine-based protocols do).
    pub open_sentinel_only: Option<&'static str>,
}

impl ProtocolTestFixture {
    /// Run every universal scenario in turn. Failure panics with
    /// scenario name + protocol context so the failing case is
    /// obvious in test output.
    pub fn run_universal_scenarios(&self) {
        self.assert_plain_text_passes_through();
        self.assert_valid_call_emits_triple();
        self.assert_unknown_tool_terminates();
        self.assert_invalid_args_terminates();
        self.assert_malformed_payload_terminates();
        self.assert_after_fatal_returns_empty();
        self.assert_utf8_text_does_not_panic();
        self.assert_chunk_split_invariance_on_valid_call();
        if self.open_sentinel_only.is_some() {
            self.assert_payload_over_limit_is_fatal();
        }
    }

    fn parser(&self) -> Box<dyn IncrementalToolCallParser> {
        (self.make_parser)(Arc::clone(&self.directory))
    }

    fn drive(
        &self,
        chunks: &[&str],
    ) -> Vec<DecodeEvent> {
        let mut p = self.parser();
        let mut events = Vec::new();
        for c in chunks {
            events.extend(p.feed(c));
        }
        events.extend(p.finish(StopReason::EndOfText));
        events
    }

    fn assert_plain_text_passes_through(&self) {
        let events = self.drive(&["hello world"]);
        let texts: String = events
            .iter()
            .filter_map(|e| match e {
                DecodeEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, "hello world", "plain-text scenario");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    fn assert_valid_call_emits_triple(&self) {
        let events = self.drive(&[self.valid_call_add_1_2]);
        let starts: Vec<&DecodeEvent> = events
            .iter()
            .filter(|e| matches!(e, DecodeEvent::ToolCallStart { .. }))
            .collect();
        assert_eq!(starts.len(), 1, "exactly one ToolCallStart for valid call");
        let DecodeEvent::ToolCallStart { name, .. } = starts[0] else {
            unreachable!()
        };
        assert_eq!(name, "add");
        let ends: Vec<&JsonValue> = events
            .iter()
            .filter_map(|e| match e {
                DecodeEvent::ToolCallEnd { args, .. } => Some(args),
                _ => None,
            })
            .collect();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0]["a"], JsonValue::from(1));
        assert_eq!(ends[0]["b"], JsonValue::from(2));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    fn assert_unknown_tool_terminates(&self) {
        let events = self.drive(&[self.unknown_tool_call]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DecodeEvent::UnknownTool { .. })),
            "unknown_tool_call must emit UnknownTool: got {events:#?}"
        );
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    fn assert_invalid_args_terminates(&self) {
        let events = self.drive(&[self.invalid_args_call]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DecodeEvent::InvalidArgs { .. })),
            "invalid_args_call must emit InvalidArgs: got {events:#?}"
        );
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    fn assert_malformed_payload_terminates(&self) {
        let events = self.drive(&[self.malformed_payload]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DecodeEvent::ParseError { .. })),
            "malformed_payload must emit ParseError: got {events:#?}"
        );
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    fn assert_after_fatal_returns_empty(&self) {
        let mut p = self.parser();
        // Drive the parser to a terminal fatal. Pair protocols emit
        // it inside `feed` (when the close sentinel arrives or never);
        // Prefix protocols emit it inside `finish` (payload drains
        // there). Either way, the fatal lands by the time the
        // following finish() returns.
        let _ = p.feed(self.malformed_payload);
        let _ = p.finish(StopReason::EndOfText);
        // After the fatal: every subsequent feed/finish returns empty.
        assert!(
            p.feed("anything else").is_empty(),
            "post-fatal feed must return empty"
        );
        assert!(
            p.finish(StopReason::EndOfText).is_empty(),
            "post-fatal finish must return empty"
        );
    }

    fn assert_utf8_text_does_not_panic(&self) {
        let _events = self.drive(&["héllo wörld 你好 "]);
        // No panic = pass.
    }

    /// Chunk-invariance lite: split the valid-call input at every
    /// possible char boundary, run through the parser, and confirm
    /// the events are the same as feeding it whole.
    fn assert_chunk_split_invariance_on_valid_call(&self) {
        let whole = format!("{:?}", self.drive(&[self.valid_call_add_1_2]));
        let text = self.valid_call_add_1_2;
        let len = text.len();
        for split in (1..len).filter(|i| text.is_char_boundary(*i)) {
            let chunked = format!("{:?}", self.drive(&[&text[..split], &text[split..]]));
            assert_eq!(
                chunked, whole,
                "chunk-split at byte {split} must yield same events"
            );
        }
    }

    fn assert_payload_over_limit_is_fatal(&self) {
        let Some(open) = self.open_sentinel_only else {
            return;
        };
        let mut p = self.parser();
        let _ = p.feed(open);
        let huge = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let events = p.feed(&huge);
        assert!(
            events.iter().any(|e| matches!(
                e,
                DecodeEvent::ParseError {
                    source: ParserError::PayloadTooLarge { .. },
                    ..
                }
            )),
            "oversize payload must emit PayloadTooLarge: got {events:#?}"
        );
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }
}

pub fn last_stop_reason(events: &[DecodeEvent]) -> StopReason {
    events
        .iter()
        .rev()
        .find_map(|e| match e {
            DecodeEvent::Stop { reason } => Some(*reason),
            _ => None,
        })
        .expect("expected a Stop event")
}

/// Shared fixture-tool builder. Every protocol's universal scenarios
/// reference the same `add(a: number, b: number)` signature, so the
/// directory itself can live here. Per-protocol modules that need
/// extra tools (mul, calculator, etc.) build their own.
pub fn add_tool() -> super::tool_spec::ToolSpec {
    use super::tool_spec::ToolSpec;
    ToolSpec::new(
        "add",
        Some("add two numbers".into()),
        serde_json::json!({
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

pub fn directory_with_add() -> Arc<ToolDirectory> {
    Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
}

/// Drive a parser through a sequence of chunks, then `finish` with
/// `EndOfText`. Returns the concatenated event list.
pub fn run(
    parser: &mut dyn IncrementalToolCallParser,
    chunks: &[&str],
) -> Vec<DecodeEvent> {
    let mut events = Vec::new();
    for c in chunks {
        events.extend(parser.feed(c));
    }
    events.extend(parser.finish(StopReason::EndOfText));
    events
}
