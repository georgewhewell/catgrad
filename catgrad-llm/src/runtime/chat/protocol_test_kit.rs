//! Universal protocol-test harness, registry-driven.
//!
//! The data each protocol contributes (wire-encoded examples for
//! valid / unknown-tool / invalid-args / malformed-payload / open
//! sentinel) lives on [`ToolCallProtocol::examples`] in
//! [`super::protocol`] — not in per-protocol test modules. This
//! module hosts ONE crate-level test fn, [`run_all_engine_scenarios`],
//! that iterates the registry and exercises the universal scenarios
//! against every protocol whose `examples` is `Some(_)`.
//!
//! Each scenario is its own free function so a failing assertion
//! still names the scenario and protocol arch in the panic message;
//! the iteration itself is just glue.
//!
//! Wire-format-specific tests (parameters-key alias, bare-object
//! form, etc.) stay in the per-protocol module — they exercise
//! dialect quirks the universal scenarios don't model.

#![cfg(test)]

use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::event::{DecodeEvent, ParserError, StopReason};
use super::parser::IncrementalToolCallParser;
use super::protocol::{ProtocolExamples, ToolCallProtocol, tool_protocol_for};
use super::sentinel_engine::MAX_TOOL_CALL_PAYLOAD_BYTES;
use super::tool_spec::ToolDirectory;

/// Architecture string + tokenizer-config pair used for the registry
/// lookup. The harness binds an `add(a: number, b: number)` tool, so
/// each protocol's `examples.valid_call_add_1_2` should resolve to a
/// valid triple.
struct ArchSpec {
    arch: &'static str,
    bos_token: Option<&'static str>,
}

impl ArchSpec {
    fn cfg(&self) -> JsonValue {
        match self.bos_token {
            Some(tok) => serde_json::json!({ "bos_token": tok }),
            None => JsonValue::Null,
        }
    }
}

/// Every architecture string the universal harness should iterate.
/// Order is stable (alphabetical within shape group) so test failures
/// reference the same row across runs.
const ARCHES: &[ArchSpec] = &[
    ArchSpec { arch: "GraniteForCausalLM",      bos_token: None },
    ArchSpec { arch: "Lfm2ForCausalLM",         bos_token: None },
    ArchSpec { arch: "MistralForCausalLM",      bos_token: None },
    ArchSpec { arch: "NemotronForCausalLM",     bos_token: None },
    ArchSpec { arch: "Olmo3ForCausalLM",        bos_token: None },
    ArchSpec { arch: "Phi4ForCausalLM",         bos_token: None },
    ArchSpec { arch: "Qwen3ForCausalLM",        bos_token: None },
    ArchSpec { arch: "Qwen3_5ForCausalLM",      bos_token: None },
    ArchSpec { arch: "SmolLM3ForCausalLM",      bos_token: None },
    ArchSpec { arch: "Gemma4ForConditionalGeneration", bos_token: None },
    // Tokenizer-fingerprinted: SmolLM2 reports `LlamaForCausalLM` and
    // is disambiguated by `<|im_start|>` BOS.
    ArchSpec { arch: "LlamaForCausalLM",        bos_token: Some("<|im_start|>") },
];

/// Crate-wide universal-scenario test. Iterates every architecture in
/// [`ARCHES`], skips any whose protocol opts out (`examples = None`),
/// and runs the full universal scenario suite against the rest. ONE
/// test fn replaces what used to be a per-protocol `passes_universal_scenarios`
/// in every engine-protocol module.
#[test]
fn run_all_engine_scenarios() {
    let mut ran = 0;
    for spec in ARCHES {
        let cfg = spec.cfg();
        let proto = tool_protocol_for(spec.arch, &cfg)
            .unwrap_or_else(|| panic!("registry lookup for arch {} returned None", spec.arch));
        let Some(examples) = proto.examples.as_ref() else {
            // Outliers (gpt_oss, llama3) opt out of the harness.
            continue;
        };
        run_scenarios_for(spec.arch, proto, examples);
        ran += 1;
    }
    assert!(
        ran >= 11,
        "expected ≥11 engine protocols to run the harness, ran {ran}"
    );
}

fn run_scenarios_for(
    arch: &str,
    proto: &'static ToolCallProtocol,
    ex: &ProtocolExamples,
) {
    let dir = directory_with_add();
    let make = |dir: Arc<ToolDirectory>| proto.make_parser(dir);

    assert_plain_text_passes_through(arch, &dir, &make);
    assert_valid_call_emits_triple(arch, &dir, &make, ex.valid_call_add_1_2);
    assert_multiple_calls_emit_sequential_indices(
        arch, &dir, &make, ex.multiple_calls_add_1_2_and_3_4,
    );
    assert_call_with_surrounding_text(arch, &dir, &make, ex.call_with_surrounding_text);
    assert_unknown_tool_terminates(arch, &dir, &make, ex.unknown_tool);
    assert_invalid_args_terminates(arch, &dir, &make, ex.invalid_args);
    assert_malformed_payload_terminates(arch, &dir, &make, ex.malformed_payload);
    assert_after_fatal_returns_empty(arch, &dir, &make, ex.malformed_payload);
    assert_utf8_text_does_not_panic(arch, &dir, &make);
    assert_chunk_split_invariance(arch, &dir, &make, ex.valid_call_add_1_2);
    assert_payload_over_limit_is_fatal(arch, &dir, &make, ex.open_sentinel_only);
}

// --- scenario helpers -----------------------------------------------

fn drive(
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    chunks: &[&str],
) -> Vec<DecodeEvent> {
    let mut p = make(Arc::clone(dir));
    let mut events = Vec::new();
    for c in chunks {
        events.extend(p.feed(c));
    }
    events.extend(p.finish(StopReason::EndOfText));
    events
}

fn assert_plain_text_passes_through(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
) {
    let events = drive(dir, make, &["hello world"]);
    let texts: String = events
        .iter()
        .filter_map(|e| match e {
            DecodeEvent::TextDelta(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, "hello world", "[{arch}] plain-text passthrough");
    assert_eq!(
        last_stop_reason(&events),
        StopReason::EndOfText,
        "[{arch}] plain-text stop reason"
    );
}

fn assert_multiple_calls_emit_sequential_indices(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    let starts: Vec<usize> = events
        .iter()
        .filter_map(|e| match e {
            DecodeEvent::ToolCallStart { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts,
        vec![0, 1],
        "[{arch}] expected two ToolCallStart events with indices [0, 1], got events {events:#?}"
    );
    let ends: Vec<&JsonValue> = events
        .iter()
        .filter_map(|e| match e {
            DecodeEvent::ToolCallEnd { args, .. } => Some(args),
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 2, "[{arch}] expected two ToolCallEnd events");
    assert_eq!(ends[0]["a"], JsonValue::from(1), "[{arch}] first call a=1");
    assert_eq!(ends[0]["b"], JsonValue::from(2), "[{arch}] first call b=2");
    assert_eq!(ends[1]["a"], JsonValue::from(3), "[{arch}] second call a=3");
    assert_eq!(ends[1]["b"], JsonValue::from(4), "[{arch}] second call b=4");
    assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
}

fn assert_call_with_surrounding_text(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    // Find the first ToolCallStart and the first TextDelta.
    let first_text = events.iter().position(|e| matches!(e, DecodeEvent::TextDelta(_)));
    let first_start = events
        .iter()
        .position(|e| matches!(e, DecodeEvent::ToolCallStart { .. }));
    let (Some(text_idx), Some(start_idx)) = (first_text, first_start) else {
        panic!(
            "[{arch}] expected at least one TextDelta and one ToolCallStart, got {events:#?}"
        );
    };
    assert!(
        text_idx < start_idx,
        "[{arch}] leading text must precede the call: events {events:#?}"
    );
    // The leading text must be non-empty (the prefix from the input).
    let DecodeEvent::TextDelta(t) = &events[text_idx] else {
        unreachable!()
    };
    assert!(
        !t.trim().is_empty(),
        "[{arch}] leading TextDelta must be non-empty, got `{t:?}`"
    );
    assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
}

fn assert_valid_call_emits_triple(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    let starts: Vec<&DecodeEvent> = events
        .iter()
        .filter(|e| matches!(e, DecodeEvent::ToolCallStart { .. }))
        .collect();
    assert_eq!(
        starts.len(),
        1,
        "[{arch}] expected exactly one ToolCallStart, got events {events:#?}"
    );
    let DecodeEvent::ToolCallStart { name, .. } = starts[0] else {
        unreachable!()
    };
    assert_eq!(name, "add", "[{arch}] valid call name");
    let end_args: Vec<&JsonValue> = events
        .iter()
        .filter_map(|e| match e {
            DecodeEvent::ToolCallEnd { args, .. } => Some(args),
            _ => None,
        })
        .collect();
    assert_eq!(end_args.len(), 1, "[{arch}] expected one ToolCallEnd");
    assert_eq!(end_args[0]["a"], JsonValue::from(1), "[{arch}] arg a=1");
    assert_eq!(end_args[0]["b"], JsonValue::from(2), "[{arch}] arg b=2");
    assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
}

fn assert_unknown_tool_terminates(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DecodeEvent::UnknownTool { .. })),
        "[{arch}] unknown_tool must emit UnknownTool: got {events:#?}"
    );
    assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
}

fn assert_invalid_args_terminates(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DecodeEvent::InvalidArgs { .. })),
        "[{arch}] invalid_args must emit InvalidArgs: got {events:#?}"
    );
    assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
}

fn assert_malformed_payload_terminates(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    input: &str,
) {
    let events = drive(dir, make, &[input]);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DecodeEvent::ParseError { .. })),
        "[{arch}] malformed must emit ParseError: got {events:#?}"
    );
    assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
}

fn assert_after_fatal_returns_empty(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    malformed: &str,
) {
    let mut p = make(Arc::clone(dir));
    let _ = p.feed(malformed);
    let _ = p.finish(StopReason::EndOfText);
    // After the fatal: every subsequent feed/finish returns empty.
    assert!(
        p.feed("anything else").is_empty(),
        "[{arch}] post-fatal feed must return empty"
    );
    assert!(
        p.finish(StopReason::EndOfText).is_empty(),
        "[{arch}] post-fatal finish must return empty"
    );
}

fn assert_utf8_text_does_not_panic(
    _arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
) {
    // No panic = pass.
    let _ = drive(dir, make, &["héllo wörld 你好 "]);
}

/// Split the valid-call input at every char boundary; same input
/// chunked any way must produce the same events.
fn assert_chunk_split_invariance(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    text: &str,
) {
    let whole = format!("{:?}", drive(dir, make, &[text]));
    let len = text.len();
    for split in (1..len).filter(|i| text.is_char_boundary(*i)) {
        let chunked = format!("{:?}", drive(dir, make, &[&text[..split], &text[split..]]));
        assert_eq!(
            chunked, whole,
            "[{arch}] chunk-split at byte {split} must yield same events"
        );
    }
}

fn assert_payload_over_limit_is_fatal(
    arch: &str,
    dir: &Arc<ToolDirectory>,
    make: &impl Fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    open: &str,
) {
    let mut p = make(Arc::clone(dir));
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
        "[{arch}] oversize payload must emit PayloadTooLarge: got {events:#?}"
    );
    assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
}

/// Look up a protocol by arch + optional BOS token (for tokenizer-
/// fingerprinted dispatch). Panics if the arch isn't registered —
/// tests should reference an arch they know exists.
pub fn protocol_for(arch: &'static str, bos_token: Option<&'static str>) -> &'static ToolCallProtocol {
    let cfg = match bos_token {
        Some(tok) => serde_json::json!({ "bos_token": tok }),
        None => JsonValue::Null,
    };
    tool_protocol_for(arch, &cfg)
        .unwrap_or_else(|| panic!("registry lookup for arch `{arch}` returned None"))
}

/// One-line parser construction for per-protocol tests:
///
/// ```ignore
/// let mut p = test_kit::make_parser_for("Qwen3ForCausalLM", None, directory);
/// ```
///
/// Replaces the ~6-line per-protocol `make_parser` shim that every
/// engine-protocol module used to ship.
pub fn make_parser_for(
    arch: &'static str,
    bos_token: Option<&'static str>,
    directory: Arc<ToolDirectory>,
) -> Box<dyn IncrementalToolCallParser> {
    protocol_for(arch, bos_token).make_parser(directory)
}

/// Centralized chunk-invariance proptest: pick a random (arch, input)
/// pair from the registry, split the input at random byte offsets,
/// and confirm the resulting events match the unsplit run. Replaces
/// per-protocol `mod proptests` blocks.
mod centralized_proptests {
    use super::*;
    use proptest::prelude::*;

    fn arch_and_input_strategy() -> impl Strategy<Value = (&'static str, &'static str)> {
        let pairs: Vec<(&'static str, &'static str)> = ARCHES
            .iter()
            .filter_map(|spec| {
                let cfg = spec.cfg();
                let proto = tool_protocol_for(spec.arch, &cfg)?;
                let ex = proto.examples?;
                Some((spec.arch, ex.interesting_inputs))
            })
            .flat_map(|(arch, inputs)| inputs.iter().map(move |inp| (arch, *inp)))
            .collect();
        // proptest's `prop::sample::select` requires at least one element.
        assert!(
            !pairs.is_empty(),
            "no (arch, input) pairs registered for chunk-invariance proptest"
        );
        proptest::sample::select(pairs)
    }

    /// Decode through the given arch's parser and accumulate into a
    /// `DecodedAssistantTurn` (or `DecodeFailure`). The accumulator
    /// collapses text deltas into a single logical text and validates
    /// the call event sequence — comparing accumulator outputs is
    /// chunk-invariant where comparing raw `DecodeEvent` sequences is
    /// not (because text-delta granularity legitimately varies with
    /// chunk boundaries).
    fn decode_through_arch(
        arch: &str,
        chunks: &[&str],
    ) -> Result<crate::runtime::chat::DecodedAssistantTurn, crate::runtime::chat::wire::DecodeFailure>
    {
        use crate::runtime::chat::AssistantTurnAccumulator;
        let cfg = ARCHES
            .iter()
            .find(|s| s.arch == arch)
            .expect("arch in proptest pair must be in ARCHES")
            .cfg();
        let proto = tool_protocol_for(arch, &cfg).expect("registered");
        let dir = directory_with_add();
        let mut parser = proto.make_parser(dir);
        let mut events = Vec::new();
        for c in chunks {
            events.extend(parser.feed(c));
        }
        events.extend(parser.finish(StopReason::EndOfText));
        let mut acc = AssistantTurnAccumulator::new();
        for ev in events {
            acc.feed(ev)?;
        }
        acc.into_turn()
    }

    proptest! {
        /// Two-way split at any byte offset must produce the same
        /// settled `DecodedAssistantTurn` as feeding the input whole.
        /// Runs over every registered (arch, input) pair.
        #[test]
        fn two_way_split_is_invariant(
            (arch, input) in arch_and_input_strategy(),
            split in 0_usize..400,
        ) {
            let split = split.min(input.len());
            let mut s = split;
            while s < input.len() && !input.is_char_boundary(s) {
                s += 1;
            }
            let whole = format!("{:?}", decode_through_arch(arch, &[input]));
            let chunked = format!("{:?}", decode_through_arch(arch, &[&input[..s], &input[s..]]));
            prop_assert_eq!(whole, chunked, "[{}] two-way split at byte {} differed", arch, s);
        }

        /// N-way split at multiple boundaries: same invariant.
        #[test]
        fn n_way_split_is_invariant(
            (arch, input) in arch_and_input_strategy(),
            mut splits in prop::collection::vec(0_usize..400, 1..5),
        ) {
            splits.sort_unstable();
            let mut clamped: Vec<usize> = Vec::with_capacity(splits.len());
            for s in splits {
                let mut s = s.min(input.len());
                while s < input.len() && !input.is_char_boundary(s) {
                    s += 1;
                }
                if clamped.last().copied().is_none_or(|prev| prev < s) {
                    clamped.push(s);
                }
            }
            let mut chunks: Vec<&str> = Vec::with_capacity(clamped.len() + 1);
            let mut last = 0;
            for s in &clamped {
                chunks.push(&input[last..*s]);
                last = *s;
            }
            chunks.push(&input[last..]);
            let whole = format!("{:?}", decode_through_arch(arch, &[input]));
            let chunked = format!("{:?}", decode_through_arch(arch, &chunks));
            prop_assert_eq!(whole, chunked, "[{}] n-way split differed", arch);
        }
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
