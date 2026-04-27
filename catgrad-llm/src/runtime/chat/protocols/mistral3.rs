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

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};
use crate::types;

const TOOL_CALL_OPEN: &str = "[TOOL_CALLS]";

/// Maximum bytes buffered after `[TOOL_CALLS]` before the parser fails
/// the call as oversized. Same rationale as the Qwen3 / LFM2 parsers:
/// large enough for any plausible structured call list, small enough
/// to bound a runaway generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Mistral 3 / Ministral 3 parser bound to the given tool
/// directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Mistral3Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the Mistral / Ministral
/// chat templates expect — the OpenAI envelope. The template's
/// `[AVAILABLE_TOOLS]` branch iterates `tools` and unpacks
/// `tool.function`, expecting `name`, `description`, and `parameters`.
pub fn render_tools(specs: &[ToolSpec]) -> JsonValue {
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

struct Mistral3Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for `[TOOL_CALLS]`.
    Outside { matcher: SentinelMatcher },
    /// `[TOOL_CALLS]` matched. Buffer everything up to EOS — there is
    /// no closing sentinel; payload boundary IS EOS.
    Inside { buffer: String },
    /// A fatal protocol error has been emitted. `feed`/`finish` return
    /// empty.
    Terminated,
}

impl Mistral3Parser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self {
            directory,
            state: State::Outside {
                matcher: SentinelMatcher::new(TOOL_CALL_OPEN),
            },
            next_index: 0,
        }
    }
}

impl IncrementalToolCallParser for Mistral3Parser {
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
                        // Transition to Inside; seed buffer with any
                        // bytes that arrived in the same chunk after
                        // the sentinel. No closing sentinel — buffer
                        // everything up to EOS.
                        self.state = State::Inside {
                            buffer: String::new(),
                        };
                        remaining = after;
                        // Fall through into the Inside branch on the
                        // next loop iteration — even if `remaining` is
                        // empty we want to enforce the size cap on the
                        // (empty) buffer, but trivially fine to just
                        // break and pick up at the next feed.
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
                State::Inside { buffer } => {
                    buffer.push_str(&remaining);
                    remaining.clear();
                    if buffer.len() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: TOOL_CALL_OPEN,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    // No closing sentinel — keep buffering until
                    // `finish()`.
                    break;
                }
                State::Terminated => break,
            }
        }
        events
    }

    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        // Take the state out so we can match by value and avoid
        // tripping the borrow checker when fatal() needs &mut self.
        let state = std::mem::replace(&mut self.state, State::Terminated);
        match state {
            State::Outside { mut matcher } => {
                let leftover = matcher.finish();
                if !leftover.is_empty() {
                    events.push(DecodeEvent::TextDelta(leftover));
                }
                events.push(DecodeEvent::Stop { reason });
                // Outside-finish is normal termination — no further
                // input expected after a finish, so leaving the parser
                // in Terminated is the right post-condition.
            }
            State::Inside { buffer } => {
                let outcome = parse_payload(&buffer, &mut self.next_index, &self.directory);
                match outcome {
                    PayloadOutcome::Calls(call_events) => {
                        events.extend(call_events);
                        events.push(DecodeEvent::Stop { reason });
                    }
                    PayloadOutcome::PartialThenFatal {
                        completed_calls,
                        fatal_event,
                    } => {
                        events.extend(completed_calls);
                        events.push(fatal_event);
                        events.push(DecodeEvent::Stop {
                            reason: StopReason::ProtocolError,
                        });
                    }
                    PayloadOutcome::Fatal(error_event) => {
                        events.push(error_event);
                        events.push(DecodeEvent::Stop {
                            reason: StopReason::ProtocolError,
                        });
                    }
                }
            }
            State::Terminated => unreachable!("checked above"),
        }
        events
    }
}

impl Mistral3Parser {
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

enum PayloadOutcome {
    /// All N calls in the payload parsed and validated. Events: `Start`,
    /// `ArgsDelta`, `End` triples in order.
    Calls(Vec<DecodeEvent>),
    /// Some prefix of the payload's calls validated, then a later call
    /// failed. The completed-call events are surfaced (so wire output
    /// is faithful to what the model produced) and the parser
    /// transitions to `Terminated`.
    PartialThenFatal {
        completed_calls: Vec<DecodeEvent>,
        fatal_event: DecodeEvent,
    },
    /// The whole payload failed before any call was validated.
    Fatal(DecodeEvent),
}

fn parse_payload(
    payload: &str,
    next_index: &mut usize,
    directory: &ToolDirectory,
) -> PayloadOutcome {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed("empty tool-call payload".into()),
        });
    }
    let value: JsonValue = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: TOOL_CALL_OPEN,
                source: ParserError::from(err),
            });
        }
    };

    // Payload may be a list of call objects or a bare object.
    let items: Vec<JsonValue> = match value {
        JsonValue::Array(items) => items,
        JsonValue::Object(_) => vec![value],
        _ => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: TOOL_CALL_OPEN,
                source: ParserError::Malformed(
                    "tool-call payload is not an object or array of objects".into(),
                ),
            });
        }
    };
    if items.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed(
                "tool-call payload contained no calls".into(),
            ),
        });
    }

    let mut events = Vec::new();
    for item in items {
        let (name, args) = match extract_call(item) {
            Ok(pair) => pair,
            Err(err) => {
                return PayloadOutcome::PartialThenFatal {
                    completed_calls: events,
                    fatal_event: DecodeEvent::ParseError {
                        sentinel: TOOL_CALL_OPEN,
                        source: err,
                    },
                };
            }
        };
        if directory.lookup(&name).is_none() {
            return PayloadOutcome::PartialThenFatal {
                completed_calls: events,
                fatal_event: DecodeEvent::UnknownTool {
                    name,
                    raw_args: args,
                },
            };
        }
        let errors = directory.validate_args(&name, &args);
        if !errors.is_empty() {
            return PayloadOutcome::PartialThenFatal {
                completed_calls: events,
                fatal_event: DecodeEvent::InvalidArgs {
                    name,
                    args,
                    errors,
                },
            };
        }
        let index = *next_index;
        *next_index += 1;
        let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
        events.push(DecodeEvent::ToolCallStart {
            index,
            name: name.clone(),
        });
        events.push(DecodeEvent::ToolCallArgsDelta {
            index,
            delta: args_text,
        });
        events.push(DecodeEvent::ToolCallEnd { index, args });
    }
    PayloadOutcome::Calls(events)
}

/// Pull `(name, arguments)` from a single Mistral call object.
///
/// The Mistral chat template emits `{"name": ..., "arguments": ...,
/// "id": "..."}`. We accept `parameters` as an alias for `arguments`
/// (consistent with the rest of the codebase) and tolerate the OpenAI-
/// legacy shape where `arguments` is a JSON-encoded string.
fn extract_call(value: JsonValue) -> Result<(String, JsonValue), ParserError> {
    let JsonValue::Object(mut obj) = value else {
        return Err(ParserError::Malformed(
            "tool-call entry is not a JSON object".into(),
        ));
    };
    let name = match obj.remove("name") {
        Some(JsonValue::String(s)) => s,
        Some(_) => {
            return Err(ParserError::Malformed(
                "tool-call `name` is not a string".into(),
            ));
        }
        None => return Err(ParserError::MissingField("name")),
    };
    let raw_args = obj
        .remove("arguments")
        .or_else(|| obj.remove("parameters"))
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));
    let args = match raw_args {
        JsonValue::String(encoded) => serde_json::from_str(&encoded).map_err(ParserError::from)?,
        other => other,
    };
    if !matches!(args, JsonValue::Object(_)) {
        return Err(ParserError::Malformed(
            "tool-call `arguments` must be an object".into(),
        ));
    }
    Ok((name, args))
}


pub fn prepare_messages(
    _specs: &[ToolSpec],
    messages: Vec<types::Message>,
) -> Vec<types::Message> {
    messages
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

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
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
    fn plain_text_passes_through_as_text_delta() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_single_call_emits_start_args_end_on_finish() {
        let mut p = Mistral3Parser::new(directory_with_add());
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"[TOOL_CALLS] [{"name":"add","parameters":{"a":1,"b":2}}]"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn arguments_as_json_encoded_string_is_accepted() {
        // OpenAI-legacy shape: `arguments` is a JSON-encoded string.
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"[TOOL_CALLS] [{"name":"add","arguments":"{\"a\":1,\"b\":2}"}]"#],
        );
        assert_eq!(events.len(), 4);
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn unknown_tool_is_terminal() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"[TOOL_CALLS] [{"name":"delete_db","arguments":{}}]"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn invalid_args_is_terminal() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"[TOOL_CALLS] [{"name":"add","arguments":{"a":"one"}}]"#],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}")
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_is_terminal() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(&mut p, &[r#"[TOOL_CALLS] not json at all"#]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal() {
        // Just `[TOOL_CALLS]` with nothing after.
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(&mut p, &["[TOOL_CALLS]"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_array_payload_is_terminal() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(&mut p, &["[TOOL_CALLS] []"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: bare JSON resembling a tool call must NOT be parsed
        // without the `[TOOL_CALLS]` sentinel.
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
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
        let mut p = Mistral3Parser::new(directory_with_add());
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
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let trigger = p.feed(r#"[TOOL_CALLS] [{"name":"missing","arguments":{}}]"#);
        // Inside still — the fatal hasn't fired yet because parse is
        // deferred to finish().
        assert!(trigger.is_empty(), "got events: {trigger:?}");
        let fin = p.finish(StopReason::EndOfText);
        assert!(matches!(&fin[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &fin[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        // Subsequent feed/finish must return empty.
        assert!(p.feed("anything").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn after_fatal_via_oversize_subsequent_feed_returns_empty() {
        // Oversize triggers fatal during feed (eagerly), which gives us
        // a different path through the state machine to test that
        // post-fatal feed/finish still return empty.
        let mut p = Mistral3Parser::new(directory_with_add());
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
    fn payload_over_limit_is_fatal() {
        let mut p = Mistral3Parser::new(directory_with_add());
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
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
    }

    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // First call is valid `add`, second is unknown — must emit the
        // first call's full triple before the UnknownTool fatal.
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(&mut p, &[r#"[TOOL_CALLS] [{"arguments":{}}]"#]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn arguments_not_object_is_terminal() {
        let mut p = Mistral3Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"[TOOL_CALLS] [{"name":"add","arguments":42}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn render_tools_produces_openai_envelope() {
        let rendered = render_tools(&[add_tool()]);
        let arr = rendered.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], json!("function"));
        assert_eq!(arr[0]["function"]["name"], json!("add"));
        assert_eq!(arr[0]["function"]["description"], json!("add two numbers"));
        assert!(arr[0]["function"]["parameters"].is_object());
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
