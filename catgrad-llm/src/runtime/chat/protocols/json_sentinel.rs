//! Generic Hermes-style sentinel-wrapped JSON tool-call parser.
//!
//! Wire format:
//!
//! ```text
//! preamble text<OPEN>{"name": "x", "arguments": {...}}</CLOSE>
//! more text<OPEN>{"name": "y", "arguments": {...}}</CLOSE>
//! ```
//!
//! Each `<OPEN>...<CLOSE>` block carries a single JSON object with at
//! minimum a `name` field; arguments arrive under either `"arguments"`
//! or `"parameters"` (the chat templates in the wild use both).
//!
//! Several architectures share this dialect (Hermes-style):
//! Qwen3 / Qwen3-MoE (sentinels `<tool_call>` / `</tool_call>`),
//! SmolLM2-Instruct (same sentinels, different system-prompt scaffolding).
//! Both protocols delegate parser construction here so they share state-
//! machine semantics, oversize-payload limits, and chunk-invariance
//! guarantees by construction.
//!
//! # Per-call atomic emission
//!
//! `<OPEN>` opens a buffering mode; only when `<CLOSE>` arrives do we
//! parse, validate, and emit the
//! `ToolCallStart` + `ToolCallArgsDelta` + `ToolCallEnd` triple as one
//! atomic unit. Call N is delivered to the client as soon as its
//! closing sentinel is seen, even while the model is still generating
//! call N+1.
//!
//! # Strict gating
//!
//! Bare JSON without a sentinel wrapper is plain text, never a tool
//! call. This avoids confusing "this looks like a tool call" content
//! inside Markdown code fences with an actual call.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory,
};

/// Maximum bytes buffered between an open and close sentinel before the
/// parser fails the call as oversized. Larger than any plausible
/// structured tool call (typical: <2 KiB; pathological: nested JSON of
/// a few KiB) and small enough that a runaway generation cannot exhaust
/// gateway memory.
pub const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a JSON-sentinel tool-call parser bound to the given tool
/// directory and open/close sentinel strings.
///
/// The parser owns the `Arc<ToolDirectory>`, so the returned
/// `Box<dyn IncrementalToolCallParser>` is `'static`.
pub fn make_parser(
    directory: Arc<ToolDirectory>,
    open: &'static str,
    close: &'static str,
) -> Box<dyn IncrementalToolCallParser> {
    Box::new(JsonSentinelParser::new(directory, open, close))
}

/// Render the bound tool list as the OpenAI-style
/// `[{"type": "function", "function": {...}}, ...]` envelope. Several
/// chat templates expect this shape under the `tools` jinja variable;
/// some (e.g. SmolLM2's bare template) ignore the variable but the
/// shape is still produced for completeness.
pub fn render_openai_tool_envelope(specs: &[crate::runtime::chat::ToolSpec]) -> JsonValue {
    JsonValue::Array(
        specs
            .iter()
            .map(|spec| {
                let mut function = JsonMap::new();
                function.insert("name".to_string(), JsonValue::String(spec.name.clone()));
                if let Some(description) = &spec.description {
                    function.insert(
                        "description".to_string(),
                        JsonValue::String(description.clone()),
                    );
                }
                function.insert("parameters".to_string(), spec.parameters.clone());
                let mut wrapper = JsonMap::new();
                wrapper.insert(
                    "type".to_string(),
                    JsonValue::String("function".to_string()),
                );
                wrapper.insert("function".to_string(), JsonValue::Object(function));
                JsonValue::Object(wrapper)
            })
            .collect(),
    )
}

struct JsonSentinelParser {
    directory: Arc<ToolDirectory>,
    open: &'static str,
    close: &'static str,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for the open sentinel.
    Outside { matcher: SentinelMatcher },
    /// Inside a tool-call block. Watching for the close sentinel; the
    /// matcher's internal buffer is the call's payload.
    Inside { matcher: SentinelMatcher },
    /// A fatal protocol error has been emitted. `feed` and `finish`
    /// return empty from this point.
    Terminated,
}

impl JsonSentinelParser {
    fn new(directory: Arc<ToolDirectory>, open: &'static str, close: &'static str) -> Self {
        Self {
            directory,
            open,
            close,
            state: State::Outside {
                matcher: SentinelMatcher::new(open),
            },
            next_index: 0,
        }
    }
}

impl IncrementalToolCallParser for JsonSentinelParser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        let mut remaining = text.to_string();
        loop {
            match &mut self.state {
                State::Outside { matcher } => {
                    matcher.push(&remaining);
                    remaining.clear();
                    if let Some((before, after)) = matcher.try_match() {
                        if !before.is_empty() {
                            events.push(DecodeEvent::TextDelta(before));
                        }
                        self.state = State::Inside {
                            matcher: SentinelMatcher::new(self.close),
                        };
                        remaining = after;
                        if remaining.is_empty() {
                            break;
                        }
                    } else {
                        let safe = matcher.flush_safe_text();
                        if !safe.is_empty() {
                            events.push(DecodeEvent::TextDelta(safe));
                        }
                        break;
                    }
                }
                State::Inside { matcher } => {
                    matcher.push(&remaining);
                    remaining.clear();
                    // Hard cap: oversized payloads are fatal — likely
                    // a runaway generation, not a real call.
                    if matcher.buffered_bytes() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: self.open,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    if let Some((payload, after)) = matcher.try_match() {
                        let index = self.next_index;
                        match parse_payload(&payload, index, self.open, &self.directory) {
                            PayloadOutcome::Call { events: call_events, count } => {
                                self.next_index += count;
                                events.extend(call_events);
                                self.state = State::Outside {
                                    matcher: SentinelMatcher::new(self.open),
                                };
                                remaining = after;
                                if remaining.is_empty() {
                                    break;
                                }
                            }
                            PayloadOutcome::Fatal(error_event) => {
                                events.extend(self.fatal(error_event));
                                return events;
                            }
                        }
                    } else {
                        // Inside, no close sentinel yet — keep buffering.
                        break;
                    }
                }
                State::Terminated => {
                    break;
                }
            }
        }
        events
    }

    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        match &mut self.state {
            State::Outside { matcher } => {
                let leftover = matcher.finish();
                if !leftover.is_empty() {
                    events.push(DecodeEvent::TextDelta(leftover));
                }
                events.push(DecodeEvent::Stop { reason });
            }
            State::Inside { .. } => {
                // Open call never closed: fatal.
                events.extend(self.fatal(DecodeEvent::ParseError {
                    sentinel: self.open,
                    source: ParserError::Unterminated,
                }));
            }
            State::Terminated => unreachable!("checked above"),
        }
        events
    }
}

impl JsonSentinelParser {
    /// Emit a fatal error event followed by `Stop { ProtocolError }`,
    /// then transition to [`State::Terminated`]. All subsequent calls
    /// to `feed` / `finish` return empty.
    fn fatal(&mut self, error_event: DecodeEvent) -> Vec<DecodeEvent> {
        self.state = State::Terminated;
        vec![
            error_event,
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError,
            },
        ]
    }
}

/// Peel a `{"type":"function","function":{...}}` envelope off a raw
/// JSON value, returning the inner call object. If the value is
/// already a bare call object, it is returned as-is. Returns `None`
/// for anything that isn't a JSON object after at most one peel.
///
/// Why this exists: small instruction-tuned models (e.g.
/// Llama-3.2-1B-Instruct) sometimes echo the *tool-spec* shape that
/// the chat template put into the prompt, instead of the response
/// shape Meta documents. Peeling lets a single parser recover both.
/// Public so per-protocol parsers (notably `llama3`) can reuse it
/// before deciding whether a JSON object is a call.
pub(super) fn peel_spec_shape_echo(value: JsonValue) -> Option<JsonMap<String, JsonValue>> {
    let mut object = value.as_object()?.clone();
    // Look for an OpenAI-spec wrapper. We peel at most once: deeper
    // nesting is almost always a hallucination, not a real call, and
    // recursing forever risks O(n) parse cost on adversarial inputs.
    let looks_like_wrapper = object
        .get("type")
        .and_then(JsonValue::as_str)
        .is_some_and(|s| s == "function")
        || object.contains_key("function");
    if !looks_like_wrapper {
        return Some(object);
    }
    if let Some(inner) = object.remove("function") {
        // The wrapper's inner `function` field can be either:
        //   - an object `{"name": "...", "arguments"|"parameters": {...}}`
        //   - a string (the raw tool name) — Llama-3.2-1B-Instruct
        //     emits this shape: `{"type":"function","function":"foo",
        //     "parameters":{...}}`. Treat the outer object's
        //     `parameters`/`arguments` as the call args and the inner
        //     string as the call name.
        match inner {
            JsonValue::Object(inner_obj) => Some(inner_obj),
            JsonValue::String(name) => {
                let mut rebuilt = JsonMap::new();
                rebuilt.insert("name".into(), JsonValue::String(name));
                if let Some(args) = object.remove("arguments") {
                    rebuilt.insert("arguments".into(), args);
                } else if let Some(params) = object.remove("parameters") {
                    rebuilt.insert("parameters".into(), params);
                }
                Some(rebuilt)
            }
            _ => None,
        }
    } else {
        // `type:"function"` but no `function` field: trust whatever
        // top-level fields are present (e.g. a `name` directly on the
        // outer object — some templates render this way).
        Some(object)
    }
}

enum PayloadOutcome {
    /// Validated calls — `events` contains `Start` / `ArgsDelta` /
    /// `End` triples for each call (contiguous, in order). `count` is
    /// the number of calls (= events.len() / 3); the parent state
    /// machine increments its `next_index` by this so subsequent
    /// `<tool_call>` blocks pick up the right index.
    Call { events: Vec<DecodeEvent>, count: usize },
    /// Anything that should not become a call: unknown name,
    /// schema-invalid args, or a parse failure.
    Fatal(DecodeEvent),
}

fn parse_payload(
    payload: &str,
    starting_index: usize,
    open: &'static str,
    directory: &ToolDirectory,
) -> PayloadOutcome {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: open,
            source: ParserError::Malformed("empty tool-call payload".into()),
        });
    }
    let value: JsonValue = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: open,
                source: ParserError::from(err),
            });
        }
    };
    // SmolLM2-Instruct's documented format wraps an array even for a
    // single call (see HuggingFaceTB/SmolLM2-1.7B-Instruct's
    // `instructions_function_calling.md`). Hermes / Qwen3 use a bare
    // object. Accept both so one parser serves both dialects.
    let raw_calls: Vec<JsonValue> = match value {
        JsonValue::Array(items) => items,
        other => vec![other],
    };
    if raw_calls.is_empty() {
        // SmolLM2's "no tool needed" answer is the literal string
        // `<tool_call>[]</tool_call>`. Treat it as a structured "no
        // calls" rather than a fatal error: emit no call events, the
        // parser keeps running, downstream handles the empty-tool-list
        // assistant turn as plain text.
        return PayloadOutcome::Call {
            events: Vec::new(),
            count: 0,
        };
    }

    let mut events = Vec::with_capacity(raw_calls.len() * 3);
    for (offset, raw) in raw_calls.into_iter().enumerate() {
        let index = starting_index + offset;
        match build_call_events(index, raw, directory) {
            CallOutcome::Ok(triple) => events.extend(triple),
            CallOutcome::Fatal(ev) => return PayloadOutcome::Fatal(ev),
        }
    }
    PayloadOutcome::Call {
        count: events.len() / 3,
        events,
    }
}

enum CallOutcome {
    /// Three events, in order: `Start`, `ArgsDelta`, `End`.
    Ok(Vec<DecodeEvent>),
    Fatal(DecodeEvent),
}

/// Extract a `(name, args)` pair from one element of the call list and
/// emit the wire-level event triple.
///
/// **Spec-shape echo handling.** Models occasionally emit the
/// OpenAI-tool-spec envelope back at us:
/// `{"type":"function","function":{"name":"x","arguments":{...}}}`.
/// llama.cpp's auto-generated grammar allows this; vLLM's Llama parser
/// does not and fails. We peel the wrapper here so the same code path
/// recovers the call. Operator note: this is forgiving — if the
/// `function` field is itself a tool-call object we recurse into it
/// once. Deeper nesting is treated as a missing-name error.
fn build_call_events(
    index: usize,
    raw: JsonValue,
    directory: &ToolDirectory,
) -> CallOutcome {
    let object = match peel_spec_shape_echo(raw) {
        Some(obj) => obj,
        None => {
            return CallOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: "<tool_call>",
                source: ParserError::Malformed(
                    "tool-call payload element is not a JSON object".into(),
                ),
            });
        }
    };
    let Some(name) = object.get("name").and_then(JsonValue::as_str) else {
        return CallOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: "<tool_call>",
            source: ParserError::MissingField("name"),
        });
    };
    // Llama's spec says `parameters`; Hermes / SmolLM2 use `arguments`.
    // Accept both with `arguments` taking precedence (matches vLLM's
    // Llama tool parser).
    let args = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .cloned()
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));

    if directory.lookup(name).is_none() {
        return CallOutcome::Fatal(DecodeEvent::UnknownTool {
            name: name.to_string(),
            raw_args: args,
        });
    }
    let errors = directory.validate_args(name, &args);
    if !errors.is_empty() {
        return CallOutcome::Fatal(DecodeEvent::InvalidArgs {
            name: name.to_string(),
            args,
            errors,
        });
    }

    let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
    CallOutcome::Ok(vec![
        DecodeEvent::ToolCallStart {
            index,
            name: name.to_string(),
        },
        DecodeEvent::ToolCallArgsDelta {
            index,
            delta: args_text,
        },
        DecodeEvent::ToolCallEnd { index, args },
    ])
}
