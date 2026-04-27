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

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};
use crate::types;

/// Prefix sentinel. No closing sentinel exists; the JSON list runs
/// until end-of-stream.
const FUNCTOOLS_PREFIX: &str = "functools";

/// Maximum bytes buffered after the `functools` prefix before the
/// parser fails as oversized. Same rationale as the Qwen3 / LFM2
/// caps: large enough for any plausible structured tool-call list,
/// small enough to bound a runaway generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Phi-3 / Phi-4-mini parser bound to the given tool
/// directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Phi4Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the Phi-4-mini chat
/// template expects.
///
/// The Phi-4-mini chat template wraps the tool list in the system
/// message as `<|tool|>{tools}<|/tool|>`, applying the value
/// directly (the template does not iterate). Microsoft's published
/// example uses a Python-callable–style description shape, but the
/// canonical OpenAI envelope (`{"type":"function","function":{...}}`)
/// is what `transformers.apply_chat_template` produces from a Python
/// callable and what the model is fine-tuned against in the
/// production deployment path. Matching that shape keeps Rust-rendered
/// prompts byte-identical to the transformers-rendered ones for the
/// same tool list.
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

struct Phi4Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for the `functools`
    /// prefix.
    Outside { matcher: SentinelMatcher },
    /// Inside the post-prefix region. There is no closing sentinel;
    /// `buffer` accumulates everything until `finish()` or until the
    /// hard size cap is hit.
    Inside { buffer: String },
    /// A fatal error has been emitted. `feed`/`finish` return empty.
    Terminated,
}

impl Phi4Parser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self {
            directory,
            state: State::Outside {
                matcher: SentinelMatcher::new(FUNCTOOLS_PREFIX),
            },
            next_index: 0,
        }
    }
}

impl IncrementalToolCallParser for Phi4Parser {
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
                        // Transition into Inside; seed buffer with the
                        // tail that came after the prefix in this same
                        // feed.
                        self.state = State::Inside {
                            buffer: String::new(),
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
                State::Inside { buffer } => {
                    buffer.push_str(&remaining);
                    remaining.clear();
                    if buffer.len() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: FUNCTOOLS_PREFIX,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    // No close sentinel — keep buffering until finish().
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
            State::Inside { buffer } => {
                let buffer = std::mem::take(buffer);
                let outcome = parse_payload(&buffer, &mut self.next_index, &self.directory);
                match outcome {
                    PayloadOutcome::Calls(call_events) => {
                        events.extend(call_events);
                        events.push(DecodeEvent::Stop { reason });
                        // Outside-finish-equivalent: the parse
                        // succeeded, so this is a normal terminal
                        // state. Mark Terminated to be defensive — no
                        // further feeds are expected after finish()
                        // anyway.
                        self.state = State::Terminated;
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

impl Phi4Parser {
    /// Emit a fatal error event followed by `Stop { ProtocolError }`,
    /// then transition to `State::Terminated`. All subsequent calls
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
    /// All N calls in the block parsed and validated. Events:
    /// `Start`, `ArgsDelta`, `End` triples in order.
    Calls(Vec<DecodeEvent>),
    /// Some prefix of the calls validated, then a later call failed.
    /// The completed-call events are surfaced and the parser
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
            sentinel: FUNCTOOLS_PREFIX,
            source: ParserError::Malformed("empty tool-call payload".into()),
        });
    }

    // Phi-4-mini's documented shape is a JSON list. We also accept a
    // bare JSON object (a single call without the surrounding `[]`)
    // because the closing-bracket-omitted shape is observed in the
    // wild and matches the lenience the qwen3 / lfm2 parsers afford
    // their own dialects.
    let value: JsonValue = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: FUNCTOOLS_PREFIX,
                source: ParserError::from(err),
            });
        }
    };

    let items = match value {
        JsonValue::Array(items) => items,
        JsonValue::Object(_) => vec![value],
        _ => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: FUNCTOOLS_PREFIX,
                source: ParserError::Malformed(
                    "tool-call payload is not an object or array of objects".into(),
                ),
            });
        }
    };

    if items.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: FUNCTOOLS_PREFIX,
            source: ParserError::Malformed("tool-call payload contained no calls".into()),
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
                        sentinel: FUNCTOOLS_PREFIX,
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
    // Accept `arguments` (canonical) or `parameters` (a common
    // model-emitted variant — same lenience as the Qwen3 / LFM2
    // parsers).
    let args = obj
        .remove("arguments")
        .or_else(|| obj.remove("parameters"))
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));
    // Some models emit arguments as a JSON-encoded string (the OpenAI
    // legacy shape). Decode if so.
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
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_single_call_emits_on_finish() {
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Phi4Parser::new(directory_with_add_and_mul());
        let events = run(
            &mut p,
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
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
    fn unknown_tool_is_terminal() {
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"functools[{"name":"delete_db","arguments":{}}]"#],
        );
        // UnknownTool, Stop{ProtocolError}
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn invalid_args_is_terminal() {
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"functools[{"name":"add","arguments":{"a":"one"}}]"#],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_is_terminal() {
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(&mut p, &[r#"functools[not json at all"#]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal() {
        // Just `functools` with no JSON before EOS.
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(&mut p, &["functools"]);
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
        let mut p = Phi4Parser::new(directory_with_add());
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
    fn partial_sentinel_split_across_feeds() {
        let mut p = Phi4Parser::new(directory_with_add());
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
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = Phi4Parser::new(directory_with_add());
        // Trigger fatal via unknown tool. This requires finish() to
        // actually commit the parse, since the close sentinel does
        // not exist.
        let mut first = p.feed(r#"functools[{"name":"x","arguments":{}}]"#);
        first.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &first[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        // Subsequent feed: empty.
        let after_feed = p.feed("any further text");
        assert!(after_feed.is_empty(), "got events: {after_feed:?}");
        // Subsequent finish: empty.
        let after_finish = p.finish(StopReason::EndOfText);
        assert!(after_finish.is_empty(), "got events: {after_finish:?}");
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = Phi4Parser::new(directory_with_add());
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("functools"));
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
        // Subsequent feed/finish: empty.
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn partial_then_fatal_emits_completed_calls_before_terminal() {
        // First call validates, second is unknown. The first triple
        // must be emitted, then UnknownTool + Stop{ProtocolError}.
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
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
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"functools{"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameters_key_accepted_as_arguments() {
        let mut p = Phi4Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"functools[{"name":"add","parameters":{"a":1,"b":2}}]"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
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
