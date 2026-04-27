//! Qwen3.5 tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble<tool_call>
//! <function=tool_name>
//! <parameter=arg_name>arg_value</parameter>
//! <parameter=other_arg>another</parameter>
//! </function>
//! </tool_call>postamble
//! ```
//!
//! The outer sentinel pair is the same as Qwen3 (`<tool_call>` /
//! `</tool_call>`), but the payload is XML-shaped rather than JSON: a
//! single `<function=NAME>...</function>` block whose body is a sequence
//! of `<parameter=KEY>VALUE</parameter>` blocks.
//!
//! Per-parameter `VALUE` is parsed with `serde_json::from_str` after
//! trimming; failures fall back to a bare string. This mirrors the
//! `parse_scalar` helper in the legacy non-streaming parser
//! (`helpers/tool_calls.rs`).
//!
//! Multiple parallel calls per generation are emitted as multiple
//! back-to-back `<tool_call>...</tool_call>` blocks (NOT multiple
//! functions inside one block); each block is one call.
//!
//! # Strict gating
//!
//! As with Qwen3: a model output that contains XML-shaped text but no
//! `<tool_call>` wrapper is plain text, never a tool call.
//!
//! # Per-call atomic emission
//!
//! Same contract as Qwen3: `<tool_call>` opens a buffer; only when
//! `</tool_call>` arrives do we parse, validate, and emit
//! `ToolCallStart` + `ToolCallArgsDelta` + `ToolCallEnd` as one atomic
//! triple. Validation failures take the `UnknownTool` / `InvalidArgs` /
//! `ParseError` paths and never emit a `ToolCallStart`.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Maximum bytes buffered between `<tool_call>` and `</tool_call>` before
/// the parser fails the call as oversized. Same rationale as Qwen3 /
/// LFM2: large enough for any plausible structured call, small enough to
/// bound a runaway generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Qwen3.5 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Qwen3_5Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the Qwen3.5 chat
/// template expects — the OpenAI-style `[{"type":"function",
/// "function":{...}}, ...]` envelope. Byte-for-byte identical to the
/// Qwen3 `render_tools` output for the same input.
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

#[allow(non_camel_case_types)]
struct Qwen3_5Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for `<tool_call>`.
    Outside { matcher: SentinelMatcher },
    /// Inside a tool-call block. Watching for `</tool_call>`; the
    /// matcher's internal buffer is the call's payload.
    Inside { matcher: SentinelMatcher },
    /// A fatal protocol error has been emitted. `feed`/`finish` return
    /// empty.
    Terminated,
}

impl Qwen3_5Parser {
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

impl IncrementalToolCallParser for Qwen3_5Parser {
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
                            matcher: SentinelMatcher::new(TOOL_CALL_CLOSE),
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
                    if matcher.buffered_bytes() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: TOOL_CALL_OPEN,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    if let Some((payload, after)) = matcher.try_match() {
                        let index = self.next_index;
                        match parse_payload(&payload, index, &self.directory) {
                            PayloadOutcome::Call(call_events) => {
                                self.next_index += 1;
                                events.extend(call_events);
                                self.state = State::Outside {
                                    matcher: SentinelMatcher::new(TOOL_CALL_OPEN),
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
                        break;
                    }
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
            State::Inside { .. } => {
                events.extend(self.fatal(DecodeEvent::ParseError {
                    sentinel: TOOL_CALL_OPEN,
                    source: ParserError::Unterminated,
                }));
            }
            State::Terminated => unreachable!("checked above"),
        }
        events
    }
}

impl Qwen3_5Parser {
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
    Call(Vec<DecodeEvent>),
    Fatal(DecodeEvent),
}

fn parse_payload(payload: &str, index: usize, directory: &ToolDirectory) -> PayloadOutcome {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed("empty tool-call payload".into()),
        });
    }

    let (name, args) = match parse_function_block(trimmed) {
        Ok(parsed) => parsed,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: TOOL_CALL_OPEN,
                source: err,
            });
        }
    };

    if directory.lookup(&name).is_none() {
        return PayloadOutcome::Fatal(DecodeEvent::UnknownTool {
            name,
            raw_args: args,
        });
    }
    let errors = directory.validate_args(&name, &args);
    if !errors.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::InvalidArgs {
            name,
            args,
            errors,
        });
    }

    let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
    PayloadOutcome::Call(vec![
        DecodeEvent::ToolCallStart {
            index,
            name: name.clone(),
        },
        DecodeEvent::ToolCallArgsDelta {
            index,
            delta: args_text,
        },
        DecodeEvent::ToolCallEnd { index, args },
    ])
}

/// Parse the body of a `<tool_call>...</tool_call>` block. Returns the
/// function name and the assembled argument object.
fn parse_function_block(payload: &str) -> Result<(String, JsonValue), ParserError> {
    const FUNCTION_OPEN: &str = "<function=";
    const FUNCTION_CLOSE: &str = "</function>";
    const PARAMETER_OPEN: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";

    let function_start = payload.find(FUNCTION_OPEN).ok_or_else(|| {
        ParserError::Malformed("tool-call payload missing <function=...> block".into())
    })?;
    let header = &payload[function_start + FUNCTION_OPEN.len()..];
    let name_end = header
        .find('>')
        .ok_or_else(|| ParserError::Malformed("unterminated <function=...> tag".into()))?;
    let name = header[..name_end].trim().to_string();
    if name.is_empty() {
        return Err(ParserError::Malformed(
            "tool call has empty function name".into(),
        ));
    }
    let body = &header[name_end + 1..];
    let body_end = body.find(FUNCTION_CLOSE).ok_or_else(|| {
        ParserError::Malformed("missing </function> in tool-call payload".into())
    })?;
    let function_body = &body[..body_end];

    let mut arguments = JsonMap::new();
    let mut rest = function_body;
    while let Some(parameter_start) = rest.find(PARAMETER_OPEN) {
        let block = &rest[parameter_start + PARAMETER_OPEN.len()..];
        let key_end = block.find('>').ok_or_else(|| {
            ParserError::Malformed("unterminated <parameter=...> tag".into())
        })?;
        let key = block[..key_end].trim().to_string();
        if key.is_empty() {
            return Err(ParserError::Malformed(
                "tool-call parameter has empty name".into(),
            ));
        }
        let value_text = &block[key_end + 1..];
        let value_end = value_text.find(PARAMETER_CLOSE).ok_or_else(|| {
            ParserError::Malformed("missing </parameter> in tool-call payload".into())
        })?;
        let value = &value_text[..value_end];
        arguments.insert(key, parse_scalar(value));
        rest = &value_text[value_end + PARAMETER_CLOSE.len()..];
    }

    Ok((name, JsonValue::Object(arguments)))
}

/// Parse a parameter value: try as JSON first (with whitespace trimmed),
/// fall back to a bare string. Mirrors the legacy
/// `helpers/tool_calls.rs::parse_scalar`.
fn parse_scalar(text: &str) -> JsonValue {
    let trimmed = text.trim();
    serde_json::from_str(trimmed).unwrap_or_else(|_| JsonValue::String(trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;

    fn calculator_tool() -> ToolSpec {
        ToolSpec::new(
            "calculator",
            Some("calculate".into()),
            json!({
                "type": "object",
                "properties": {
                    "lhs": { "type": "number" },
                    "rhs": { "type": "number" },
                    "op":  { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                },
                "required": ["lhs", "rhs", "op"],
                "additionalProperties": false,
            }),
        )
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

    fn greet_tool() -> ToolSpec {
        ToolSpec::new(
            "greet",
            None,
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "times": { "type": "number" },
                },
                "required": ["name"],
                "additionalProperties": false,
            }),
        )
    }

    fn directory_with_calculator() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![calculator_tool()]).unwrap())
    }

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
    }

    fn directory_with_greet() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![greet_tool()]).unwrap())
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
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_call_emits_start_args_end() {
        let mut p = Qwen3_5Parser::new(directory_with_greet());
        let events = run(
            &mut p,
            &[
                "<tool_call>\n<function=greet>\n<parameter=name>\"Alice\"</parameter>\n</function>\n</tool_call>",
            ],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "greet"
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::ToolCallArgsDelta { index: 0, .. }
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2])
        };
        assert_eq!(args, &json!({ "name": "Alice" }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_call_with_multiple_parameters() {
        // String + number args; confirms parse_scalar's JSON-promotion of
        // numeric and quoted-string values.
        let mut p = Qwen3_5Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[
                "<tool_call>\n<function=calculator>\n<parameter=lhs>1353785</parameter>\n<parameter=rhs>790489</parameter>\n<parameter=op>\"div\"</parameter>\n</function>\n</tool_call>",
            ],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["lhs"], json!(1353785));
        assert_eq!(args["rhs"], json!(790489));
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn parameter_value_falls_back_to_bare_string() {
        // Bare `div` (no quotes) — JSON parse fails, fall back to string.
        let mut p = Qwen3_5Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[
                "<tool_call>\n<function=calculator>\n<parameter=lhs>1</parameter>\n<parameter=rhs>2</parameter>\n<parameter=op>div</parameter>\n</function>\n</tool_call>",
            ],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn multiple_calls_in_sequence() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<tool_call>\n<function=add>\n<parameter=a>1</parameter>\n<parameter=b>2</parameter>\n</function>\n</tool_call>",
                "<tool_call>\n<function=add>\n<parameter=a>3</parameter>\n<parameter=b>4</parameter>\n</function>\n</tool_call>",
            ],
        );
        // Start(0), ArgsDelta(0), End(0), Start(1), ArgsDelta(1), End(1), Stop
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
    fn text_then_call_then_text_in_single_feed() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "first <tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> last",
            ],
        );
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
    fn unknown_tool_is_terminal_with_protocol_error() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<tool_call><function=delete_db></function></tool_call>",
            ],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal_with_protocol_error() {
        let mut p = Qwen3_5Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[
                "<tool_call><function=calculator><parameter=lhs>1</parameter><parameter=rhs>2</parameter><parameter=op>\"bogus\"</parameter></function></tool_call>",
            ],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}")
        };
        assert_eq!(name, "calculator");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_xml_is_terminal_with_protocol_error() {
        // <parameter=a>1 with no closing </parameter> at all — the
        // remaining body has no `</parameter>` after the parameter
        // header, so parsing the param value fails.
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<tool_call><function=add><parameter=a>1</function></tool_call>",
            ],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_parameter_tag_is_terminal_with_protocol_error() {
        // <parameter=a missing its closing '>' — this hits the
        // "unterminated <parameter=...> tag" branch.
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<tool_call><function=add><parameter=a 1</parameter></function></tool_call>",
            ],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_function_block_is_terminal_with_protocol_error() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<tool_call>just some text without a function block</tool_call>"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal_with_protocol_error() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(&mut p, &["<tool_call></tool_call>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<tool_call><function=add><parameter=a>1</parameter>"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}")
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        // Should not have emitted the partial sentinel as text.
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains("<tool_c"),
            _ => true,
        }));
        events.extend(p.feed(
            "all><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> done",
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
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_",
        ));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_xml_without_sentinel_is_plain_text() {
        // Critical: bare XML resembling a tool call must NOT be parsed.
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "Here is some XML: <function=add><parameter=a>1</parameter></function>",
            ],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "Here is some XML: <function=add><parameter=a>1</parameter></function>"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let first = p.feed("<tool_call><function=delete_db></function></tool_call>");
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &first[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        assert!(p.feed("anything").is_empty());
        assert!(
            p.feed("<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>")
                .is_empty()
        );
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = Qwen3_5Parser::new(directory_with_add());
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<tool_call>"));
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

    // --- shape unit tests (do not exercise the state machine) ---

    #[test]
    fn parse_scalar_promotes_json_values() {
        assert_eq!(parse_scalar("42"), json!(42));
        assert_eq!(parse_scalar("3.14"), json!(3.14));
        assert_eq!(parse_scalar("true"), json!(true));
        assert_eq!(parse_scalar("null"), json!(null));
        assert_eq!(parse_scalar("\"hi\""), json!("hi"));
        assert_eq!(parse_scalar("[1, 2, 3]"), json!([1, 2, 3]));
        assert_eq!(parse_scalar(r#"{"a": 1}"#), json!({"a": 1}));
    }

    #[test]
    fn parse_scalar_falls_back_to_string() {
        assert_eq!(parse_scalar("div"), json!("div"));
        assert_eq!(parse_scalar("  hello world  "), json!("hello world"));
        // Unquoted multi-word remains a string.
        assert_eq!(parse_scalar("not json"), json!("not json"));
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
            // single valid call
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>",
            // call with newlines (the canonical chat-template shape)
            "<tool_call>\n<function=add>\n<parameter=a>1</parameter>\n<parameter=b>2</parameter>\n</function>\n</tool_call>",
            // call with surrounding text
            "prefix <tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call> suffix",
            // two calls in sequence
            "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call><tool_call><function=add><parameter=a>3</parameter><parameter=b>4</parameter></function></tool_call>",
            // sentinel-shaped text that isn't a sentinel
            "the docs say <tool_call but it's just text",
            // unknown tool (fatal — chunk-invariance still holds)
            "<tool_call><function=missing></function></tool_call>",
            // bare-string parameter value falling back through parse_scalar
            "<tool_call><function=add><parameter=a>not_json</parameter><parameter=b>2</parameter></function></tool_call>",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..8,
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
            input_idx in 0_usize..8,
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
