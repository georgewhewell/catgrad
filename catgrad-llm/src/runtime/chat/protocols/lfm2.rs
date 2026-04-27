//! LFM2 / LFM2.5 tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble<|tool_call_start|>[calculator(lhs=1, rhs=2, op="div")]<|tool_call_end|>postamble
//! ```
//!
//! The block between `<|tool_call_start|>` and `<|tool_call_end|>` may
//! carry one of two payload shapes:
//!
//! - **Pythonic** (LiquidAI default): a Python list literal of calls,
//!   `[name1(k=v, ...), name2(k=v, ...)]`. The outer `[...]` is also
//!   accepted as omitted (a single bare call).
//! - **JSON** (alternate, requested by system prompt): `[{"name": "x",
//!   "arguments": {...}}]` — same shape as Qwen3 but wrapped in a list,
//!   and a bare object outside the list is also accepted.
//!
//! The dispatch is by sniffing the first non-whitespace byte: `[` or `{`
//! routes to JSON; anything else routes to Pythonic. If JSON parsing
//! fails on a payload that opened with `[` or `{`, the parser falls back
//! to Pythonic — model outputs occasionally mix the two (e.g. a JSON-
//! looking dict literal that uses single quotes).
//!
//! # Multiple calls per block
//!
//! Both payload shapes can carry multiple calls. The parser emits one
//! `ToolCallStart` / `ToolCallArgsDelta` / `ToolCallEnd` triple per
//! parsed call, with strictly increasing indices, atomically (no
//! interleaving). If call N+1 fails validation, calls 0..N have already
//! been emitted and the fatal event terminates the parser.
//!
//! # References
//!
//! - LFM2.5 chat template:
//!   <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/chat_template.jinja>
//! - SGLang reference parser:
//!   <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/lfm2_detector.py>

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};
use crate::types;

const TOOL_CALL_OPEN: &str = "<|tool_call_start|>";
const TOOL_CALL_CLOSE: &str = "<|tool_call_end|>";

/// Maximum bytes buffered between sentinels before the parser fails the
/// call as oversized. Same rationale as the Qwen3 parser: large enough
/// for any plausible structured call, small enough to bound a runaway
/// generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct an LFM2 / LFM2.5 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Lfm2Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the LFM2.5 chat
/// template expects.
///
/// LFM2.5's template applies `tojson` to each tool item directly (no
/// envelope), but the canonical shape that `transformers.apply_chat_template`
/// produces from a Python callable — and that the LFM2.5 family was
/// fine-tuned against — is the OpenAI envelope: `{"type":"function",
/// "function":{...}}`. Matching that shape keeps Rust-rendered prompts
/// byte-identical to transformers-rendered prompts for the same tool
/// list.
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

struct Lfm2Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any block. Watching for `<|tool_call_start|>`.
    Outside { matcher: SentinelMatcher },
    /// Inside a block. Watching for `<|tool_call_end|>`; the matcher's
    /// internal buffer is the call's payload.
    Inside { matcher: SentinelMatcher },
    /// A fatal error has been emitted. `feed`/`finish` return empty.
    Terminated,
}

impl Lfm2Parser {
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

impl IncrementalToolCallParser for Lfm2Parser {
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
                        let outcome =
                            parse_payload(&payload, &mut self.next_index, &self.directory);
                        match outcome {
                            PayloadOutcome::Calls(call_events) => {
                                events.extend(call_events);
                                self.state = State::Outside {
                                    matcher: SentinelMatcher::new(TOOL_CALL_OPEN),
                                };
                                remaining = after;
                                if remaining.is_empty() {
                                    break;
                                }
                            }
                            PayloadOutcome::PartialThenFatal {
                                completed_calls,
                                fatal_event,
                            } => {
                                events.extend(completed_calls);
                                events.extend(self.fatal(fatal_event));
                                return events;
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

impl Lfm2Parser {
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

    let parsed = parse_payload_calls(trimmed);
    let calls = match parsed {
        Ok(c) => c,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: TOOL_CALL_OPEN,
                source: err,
            });
        }
    };
    if calls.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed(
                "tool-call payload contained no calls".into(),
            ),
        });
    }

    let mut events = Vec::new();
    for (name, args) in calls {
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

/// Dispatch the trimmed payload to JSON or Pythonic parsing. JSON is
/// tried first when the payload starts with `[` or `{`; on JSON parse
/// failure we fall back to Pythonic so a malformed-JSON payload that
/// happened to start with `[` is still parseable as e.g.
/// `[func(a='x')]`. Returns a vec of `(name, arguments)` pairs.
fn parse_payload_calls(text: &str) -> Result<Vec<(String, JsonValue)>, ParserError> {
    let first = text.chars().next();
    match first {
        Some('[') | Some('{') => match parse_json_calls(text) {
            Ok(calls) => Ok(calls),
            Err(_) => parse_python_calls(text),
        },
        _ => parse_python_calls(text),
    }
}

fn parse_json_calls(text: &str) -> Result<Vec<(String, JsonValue)>, ParserError> {
    let value: JsonValue = serde_json::from_str(text).map_err(ParserError::from)?;
    let items = match value {
        JsonValue::Array(items) => items,
        JsonValue::Object(_) => vec![value],
        _ => {
            return Err(ParserError::Malformed(
                "JSON tool-call payload is not an object or array of objects".into(),
            ));
        }
    };
    items
        .into_iter()
        .map(json_call_from_value)
        .collect::<Result<Vec<_>, _>>()
}

fn json_call_from_value(value: JsonValue) -> Result<(String, JsonValue), ParserError> {
    let JsonValue::Object(mut obj) = value else {
        return Err(ParserError::Malformed(
            "JSON tool-call entry is not an object".into(),
        ));
    };
    let name = match obj.remove("name") {
        Some(JsonValue::String(s)) => s,
        Some(_) => {
            return Err(ParserError::Malformed(
                "JSON tool-call `name` is not a string".into(),
            ));
        }
        None => return Err(ParserError::MissingField("name")),
    };
    let args = obj
        .remove("arguments")
        .or_else(|| obj.remove("parameters"))
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));
    // Accept arguments as a JSON-encoded string (the OpenAI legacy shape).
    let args = match args {
        JsonValue::String(encoded) => serde_json::from_str(&encoded).map_err(ParserError::from)?,
        other => other,
    };
    if !matches!(args, JsonValue::Object(_)) {
        return Err(ParserError::Malformed(
            "JSON tool-call `arguments` must be an object".into(),
        ));
    }
    Ok((name, args))
}

/// Strip a single matching pair of outer brackets from a Pythonic
/// payload. Only strips when the FIRST byte is `[` and the LAST byte is
/// `]` after trimming — so `[a(),b()]` → `a(),b()`, but `a(x=[1,2])`
/// keeps its inner brackets intact.
fn strip_outer_list(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.len() >= 2 {
        trimmed[1..trimmed.len() - 1].trim()
    } else {
        trimmed
    }
}

fn parse_python_calls(text: &str) -> Result<Vec<(String, JsonValue)>, ParserError> {
    let inner = strip_outer_list(text);
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner, ',')?
        .into_iter()
        .map(parse_python_call)
        .collect()
}

fn parse_python_call(text: &str) -> Result<(String, JsonValue), ParserError> {
    let text = text.trim();
    let open_paren = text
        .find('(')
        .ok_or_else(|| ParserError::Malformed(format!("missing `(` in tool call: {text}")))?;
    let close_paren = text
        .rfind(')')
        .ok_or_else(|| ParserError::Malformed(format!("missing `)` in tool call: {text}")))?;
    if close_paren <= open_paren {
        return Err(ParserError::Malformed(format!(
            "malformed tool call (mismatched parens): {text}"
        )));
    }
    let name = text[..open_paren].trim();
    if name.is_empty() {
        return Err(ParserError::Malformed(
            "tool call has empty function name".into(),
        ));
    }

    let args_text = text[open_paren + 1..close_paren].trim();
    let mut arguments = JsonMap::new();
    if !args_text.is_empty() {
        for arg in split_top_level(args_text, ',')? {
            let eq = find_top_level_char(arg, '=').ok_or_else(|| {
                ParserError::Malformed(format!("missing `=` in tool argument: {arg}"))
            })?;
            let key = arg[..eq].trim();
            if key.is_empty() {
                return Err(ParserError::Malformed(format!(
                    "tool argument has empty name: {arg}"
                )));
            }
            let value = arg[eq + 1..].trim();
            arguments.insert(key.to_string(), parse_python_value(value)?);
        }
    }
    Ok((name.to_string(), JsonValue::Object(arguments)))
}

fn parse_python_value(text: &str) -> Result<JsonValue, ParserError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(ParserError::Malformed("empty argument value".into()));
    }
    if text.len() >= 2
        && ((text.starts_with('"') && text.ends_with('"'))
            || (text.starts_with('\'') && text.ends_with('\'')))
    {
        return Ok(JsonValue::String(parse_python_string(text)?));
    }
    match text {
        "True" => Ok(JsonValue::Bool(true)),
        "False" => Ok(JsonValue::Bool(false)),
        "None" => Ok(JsonValue::Null),
        _ => match serde_json::from_str(text) {
            Ok(v) => Ok(v),
            // Fallback: treat as a bare identifier or string-without-quotes.
            // This is intentionally permissive — matches the reference SGLang
            // parser's behavior on unquoted enum-like values.
            Err(_) => Ok(JsonValue::String(text.to_string())),
        },
    }
}

fn parse_python_string(text: &str) -> Result<String, ParserError> {
    let quote = text
        .chars()
        .next()
        .ok_or_else(|| ParserError::Malformed("empty Python string".into()))?;
    if !text.ends_with(quote) || text.len() < 2 {
        return Err(ParserError::Malformed("unterminated Python string".into()));
    }
    let inner = &text[1..text.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let Some(escaped) = chars.next() else {
                return Err(ParserError::Malformed(
                    "unterminated escape in Python string".into(),
                ));
            };
            out.push(match escaped {
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

/// Split `text` on `separator`, respecting nesting in brackets/parens/
/// braces and quoted strings. Empty parts (e.g. trailing comma) are
/// dropped.
fn split_top_level(text: &str, separator: char) -> Result<Vec<&str>, ParserError> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth_paren = 0usize;
    let mut depth_bracket = 0usize;
    let mut depth_brace = 0usize;
    let mut in_quote: Option<char> = None;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if let Some(quote) = in_quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == quote {
                in_quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => in_quote = Some(ch),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            _ if ch == separator
                && depth_paren == 0
                && depth_bracket == 0
                && depth_brace == 0 =>
            {
                let part = text[start..idx].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    if in_quote.is_some() || depth_paren != 0 || depth_bracket != 0 || depth_brace != 0 {
        return Err(ParserError::Malformed(format!(
            "unterminated tool-call expression: {text}"
        )));
    }
    let part = text[start..].trim();
    if !part.is_empty() {
        parts.push(part);
    }
    Ok(parts)
}

fn find_top_level_char(text: &str, needle: char) -> Option<usize> {
    let mut depth_paren = 0usize;
    let mut depth_bracket = 0usize;
    let mut depth_brace = 0usize;
    let mut in_quote: Option<char> = None;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if let Some(quote) = in_quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == quote {
                in_quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => in_quote = Some(ch),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            _ if ch == needle
                && depth_paren == 0
                && depth_bracket == 0
                && depth_brace == 0 =>
            {
                return Some(idx);
            }
            _ => {}
        }
    }
    None
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

    fn directory_with_calculator() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![calculator_tool()]).unwrap())
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
        let mut p = Lfm2Parser::new(directory_with_calculator());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn pythonic_call_with_outer_list() {
        let mut p = Lfm2Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[
                "<|tool_call_start|>[calculator(lhs=1353785, rhs=790489, op=\"div\")]<|tool_call_end|>",
            ],
        );
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "calculator"
        ));
        let DecodeEvent::ToolCallEnd { index: 0, args } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args["lhs"], json!(1353785));
        assert_eq!(args["rhs"], json!(790489));
        assert_eq!(args["op"], json!("div"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn pythonic_call_without_outer_list() {
        // Bare `name(args)` (no enclosing `[...]`) is also accepted.
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|tool_call_start|>add(a=1, b=2)<|tool_call_end|>"],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!()
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn pythonic_single_quoted_string_arg() {
        let mut p = Lfm2Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &["<|tool_call_start|>[calculator(lhs=1, rhs=2, op='div')]<|tool_call_end|>"],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got {:?}", events)
        };
        assert_eq!(args["op"], json!("div"));
    }

    #[test]
    fn json_call_array_payload() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}}]<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!()
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn json_call_bare_object_payload() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call_start|>{"name":"add","arguments":{"a":1,"b":2}}<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn json_call_parameters_key_accepted() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"<|tool_call_start|>[{"name":"add","parameters":{"a":1,"b":2}}]<|tool_call_end|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
    }

    #[test]
    fn pythonic_multiple_calls_emit_sequential_indices() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|tool_call_start|>[add(a=1,b=2), add(a=3,b=4)]<|tool_call_end|>",
            ],
        );
        // Two triples + Stop = 7 events.
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn json_multiple_calls_emit_sequential_indices() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"add","arguments":{"a":3,"b":4}}]<|tool_call_end|>"#,
            ],
        );
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
    }

    #[test]
    fn calls_in_separate_blocks_keep_index_sequential() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|>",
                "<|tool_call_start|>[add(a=3,b=4)]<|tool_call_end|>",
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, .. }
        ));
    }

    #[test]
    fn text_then_call_then_text_in_single_feed() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "first <|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|> last",
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "first "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " last"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unknown_tool_is_terminal() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|tool_call_start|>[delete_db()]<|tool_call_end|>"],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn invalid_args_is_terminal() {
        let mut p = Lfm2Parser::new(directory_with_calculator());
        let events = run(
            &mut p,
            &[r#"<|tool_call_start|>[calculator(lhs=1, rhs=2, op="bogus")]<|tool_call_end|>"#],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}")
        };
        assert_eq!(name, "calculator");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn second_call_unknown_after_first_succeeds_emits_first_then_fatal() {
        // Validates the partial-then-fatal contract: calls 0..N validated,
        // call N+1 is unknown — events must include 0..N's full triples
        // before the UnknownTool/Stop pair.
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|tool_call_start|>[add(a=1,b=2), delete_db()]<|tool_call_end|>",
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
    fn malformed_pythonic_payload_is_terminal() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|tool_call_start|>add(a=<|tool_call_end|>"],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(&mut p, &["<|tool_call_start|><|tool_call_end|>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_block_at_eos_is_terminal() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(&mut p, &["<|tool_call_start|>[add(a=1,b=2)"]);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError")
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn raw_pythonic_without_sentinel_is_plain_text() {
        // No tool-call sentinel = no parsing, even if the text looks
        // exactly like a Pythonic call. Matches the Qwen3 strict-gating
        // rule.
        let mut p = Lfm2Parser::new(directory_with_add());
        let events = run(&mut p, &["here is some text: add(a=1, b=2)"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "here is some text: add(a=1, b=2)"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <|tool_c"));
        events.extend(p.feed("all_start|>[add(a=1,b=2)]<|tool_call_end|> done"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "preamble "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " done"
        ));
    }

    #[test]
    fn partial_close_sentinel_split_across_feeds() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("<|tool_call_start|>[add(a=1,b=2)]<|tool_call_"));
        assert!(events.is_empty(), "got events early: {events:?}");
        events.extend(p.feed("end|>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let first = p.feed("<|tool_call_start|>[delete_db()]<|tool_call_end|>");
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(p.feed("anything").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = Lfm2Parser::new(directory_with_add());
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<|tool_call_start|>"));
        events.extend(p.feed(&oversize));
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(
            source,
            ParserError::PayloadTooLarge { limit_bytes }
                if *limit_bytes == MAX_TOOL_CALL_PAYLOAD_BYTES
        ));
    }

    // --- payload-shape unit tests (do not exercise the state machine) ---

    #[test]
    fn parse_python_call_extracts_keyword_arguments() {
        let (name, args) =
            parse_python_call("calculator(lhs=1, rhs=2, op=\"div\")").unwrap();
        assert_eq!(name, "calculator");
        assert_eq!(args["lhs"], json!(1));
        assert_eq!(args["op"], json!("div"));
    }

    #[test]
    fn parse_python_value_handles_python_literals() {
        assert_eq!(parse_python_value("True").unwrap(), json!(true));
        assert_eq!(parse_python_value("False").unwrap(), json!(false));
        assert_eq!(parse_python_value("None").unwrap(), json!(null));
        assert_eq!(parse_python_value("'hi'").unwrap(), json!("hi"));
        assert_eq!(parse_python_value("\"hi\"").unwrap(), json!("hi"));
        assert_eq!(parse_python_value("42").unwrap(), json!(42));
        assert_eq!(parse_python_value("3.14").unwrap(), json!(3.14));
    }

    #[test]
    fn parse_python_value_unquoted_string_falls_back() {
        // Bare `div` (no quotes) — JSON parse fails, fall back to string.
        assert_eq!(parse_python_value("div").unwrap(), json!("div"));
    }

    #[test]
    fn split_top_level_respects_nesting() {
        assert_eq!(
            split_top_level("a, b(c, d), e", ',').unwrap(),
            vec!["a", "b(c, d)", "e"]
        );
        assert_eq!(
            split_top_level("a=[1, 2], b={'x': 3}", ',').unwrap(),
            vec!["a=[1, 2]", "b={'x': 3}"]
        );
    }

    #[test]
    fn split_top_level_rejects_unterminated() {
        assert!(split_top_level("a, b(c, d", ',').is_err());
        assert!(split_top_level("a, 'unclosed", ',').is_err());
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
            // Pythonic with outer list
            "<|tool_call_start|>[add(a=1,b=2)]<|tool_call_end|>",
            // Pythonic without outer list
            "<|tool_call_start|>add(a=1,b=2)<|tool_call_end|>",
            // JSON array form
            r#"<|tool_call_start|>[{"name":"add","arguments":{"a":1,"b":2}}]<|tool_call_end|>"#,
            // JSON bare object form
            r#"<|tool_call_start|>{"name":"add","arguments":{"a":1,"b":2}}<|tool_call_end|>"#,
            // Two pythonic calls in one block
            "<|tool_call_start|>[add(a=1,b=2), add(a=3,b=4)]<|tool_call_end|>",
            // Sentinel-shaped text that isn't a sentinel
            "the docs say <|tool_call_start but it's just text",
            // Call wrapped in surrounding text
            "prefix <|tool_call_start|>add(a=1,b=2)<|tool_call_end|> suffix",
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
