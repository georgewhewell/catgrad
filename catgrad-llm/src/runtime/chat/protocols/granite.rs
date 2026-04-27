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

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};
use crate::types;

/// Granite 3.x prefix sentinel. Once seen, every subsequent byte belongs
/// to the tool-call payload (a JSON list of calls). There is no closing
/// sentinel — the payload terminates at EOS.
const TOOL_CALL_OPEN: &str = "<|tool_call|>";

/// Maximum bytes buffered inside an open `<|tool_call|>` block before
/// the parser fails the call as oversized. Same rationale as the qwen3
/// and lfm2 parsers: large enough for any plausible structured call,
/// small enough to bound a runaway generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Granite 3.x parser bound to the given tool directory.
///
/// The parser owns the `Arc<ToolDirectory>`, so the returned
/// `Box<dyn IncrementalToolCallParser>` is `'static`.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(GraniteParser::new(directory))
}

/// Render the bound tool list into the JSON shape the Granite 3.x chat
/// template expects.
///
/// The Granite chat template emits the bound `available_tools` value
/// directly via `tojson`, with no envelope. To stay byte-identical to
/// `transformers.apply_chat_template` for the same tool list — which is
/// what the model was fine-tuned against — we render the OpenAI
/// `{"type":"function","function":{...}}` envelope here, since that is
/// the canonical shape `transformers` produces from a Python tool
/// callable.
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

struct GraniteParser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for `<|tool_call|>`.
    Outside { matcher: SentinelMatcher },
    /// Inside a tool-call block. There is no closing sentinel for
    /// Granite — every subsequent byte is appended to `payload` until
    /// `finish()` is called.
    Inside { payload: String },
    /// A fatal protocol error has been emitted. `feed` and `finish`
    /// return empty from this point.
    Terminated,
}

impl GraniteParser {
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

impl IncrementalToolCallParser for GraniteParser {
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
                        let mut payload = String::new();
                        payload.push_str(&after);
                        // Eager size check: a payload that opened with
                        // a huge chunk in the same feed must fail now,
                        // not silently buffer.
                        if payload.len() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                            return self.fatal(DecodeEvent::ParseError {
                                sentinel: TOOL_CALL_OPEN,
                                source: ParserError::PayloadTooLarge {
                                    limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                                },
                            });
                        }
                        self.state = State::Inside { payload };
                        // No close sentinel to scan for — break and wait
                        // for the next feed (or `finish`).
                        break;
                    } else {
                        let safe = matcher.flush_safe_text();
                        if !safe.is_empty() {
                            events.push(DecodeEvent::TextDelta(safe));
                        }
                        break;
                    }
                }
                State::Inside { payload } => {
                    payload.push_str(&remaining);
                    remaining.clear();
                    if payload.len() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: TOOL_CALL_OPEN,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    // Granite has no close sentinel — keep buffering
                    // until `finish()`.
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
        match &mut self.state {
            State::Outside { matcher } => {
                let leftover = matcher.finish();
                if !leftover.is_empty() {
                    events.push(DecodeEvent::TextDelta(leftover));
                }
                events.push(DecodeEvent::Stop { reason });
            }
            State::Inside { payload } => {
                let payload = std::mem::take(payload);
                let outcome = parse_payload(&payload, &mut self.next_index, &self.directory);
                match outcome {
                    PayloadOutcome::Calls(call_events) => {
                        events.extend(call_events);
                        events.push(DecodeEvent::Stop { reason });
                        // Successful end-of-stream — leave state as-is;
                        // no subsequent feed/finish is expected anyway.
                    }
                    PayloadOutcome::PartialThenFatal {
                        completed_calls,
                        fatal_event,
                    } => {
                        events.extend(completed_calls);
                        events.extend(self.fatal(fatal_event));
                    }
                    PayloadOutcome::Fatal(error_event) => {
                        events.extend(self.fatal(error_event));
                    }
                }
            }
            State::Terminated => unreachable!("checked above"),
        }
        events
    }
}

impl GraniteParser {
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
    /// All N calls in the block parsed and validated. Events: `Start`,
    /// `ArgsDelta`, `End` triples in order.
    Calls(Vec<DecodeEvent>),
    /// Some prefix of the block's calls validated, then a later call
    /// failed. The completed-call events are surfaced (so wire output
    /// is faithful to what the model produced) and the parser
    /// transitions to `Terminated`.
    PartialThenFatal {
        completed_calls: Vec<DecodeEvent>,
        fatal_event: DecodeEvent,
    },
    /// The whole block failed before any call was validated.
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

    let items = match value {
        JsonValue::Array(items) => items,
        // Bare object is accepted as a single call — some Granite
        // checkpoints emit `<|tool_call|>{"name": ..., ...}` directly
        // without the enclosing list.
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
        let (name, args) = match call_from_value(item) {
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

fn call_from_value(value: JsonValue) -> Result<(String, JsonValue), ParserError> {
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
    let args = obj
        .remove("arguments")
        .or_else(|| obj.remove("parameters"))
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));
    // Accept arguments encoded as a JSON string (the OpenAI legacy
    // shape) by re-parsing.
    let args = match args {
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
    fn plain_text_passes_through_as_text_delta() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
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
    fn unknown_tool_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"delete_db","arguments":{}}]"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn invalid_args_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call|>[{"name":"add","arguments":{"a":"one"}}]"#],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_payload_is_terminal() {
        let mut p = GraniteParser::new(directory_with_add());
        let events = run(&mut p, &["<|tool_call|>not json at all"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
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
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = GraniteParser::new(directory_with_add());
        let mut events = p.feed(r#"<|tool_call|>[{"name":"missing","arguments":{}}]"#);
        // No close sentinel — no events yet.
        assert!(events.is_empty(), "got events early: {events:?}");
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        assert!(p.feed("anything").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = GraniteParser::new(directory_with_add());
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<|tool_call|>"));
        events.extend(p.feed(&oversize));
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
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
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
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

    #[test]
    fn utf8_multibyte_text_does_not_panic() {
        let mut p = GraniteParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("héllo "));
        events.extend(p.feed(r#"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"#));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "héllo ");
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
