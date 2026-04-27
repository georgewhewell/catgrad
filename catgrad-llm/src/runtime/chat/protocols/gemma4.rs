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

use std::sync::Arc;

use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};

const TOOL_CALL_OPEN: &str = "<|tool_call>";
const TOOL_CALL_CLOSE: &str = "<tool_call|>";
const STRING_QUOTE: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

/// Maximum bytes buffered between `<|tool_call>` and `<tool_call|>`
/// before the parser fails the call as oversized. See Qwen3 protocol
/// for rationale — same limit, same threat model.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Gemma 4 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Gemma4Parser::new(directory))
}

/// Render the bound tool list into the JSON shape Gemma 4's chat
/// template expects. The template's `format_function_declaration`
/// macro reads `tool['function']['name']`,
/// `tool['function']['description']`, and `tool['function']['parameters']`
/// — same OpenAI-style envelope Qwen3 uses, so the shape is shared.
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

struct Gemma4Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    Outside { matcher: SentinelMatcher },
    Inside { matcher: SentinelMatcher },
    Terminated,
}

impl Gemma4Parser {
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

impl IncrementalToolCallParser for Gemma4Parser {
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

impl Gemma4Parser {
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
        return malformed("empty tool-call payload");
    }
    // HF discussions #20 / #55 on google/gemma-4-*-it: an outdated
    // chat-template revision emits `<|tool_call>{{...JSON...}}<tool_call|>`
    // (a doubled-brace JSON literal) instead of the bare-key form. Detect
    // it before the `call:` strip so the operator gets a useful hint
    // pointing at the upstream issue.
    if trimmed.starts_with("{{") {
        return malformed(
            "tool-call payload has double-braced JSON body — \
             the chat template is an outdated revision (see \
             huggingface.co/google/gemma-4-*-it discussions #20/#55)",
        );
    }
    let Some(rest) = trimmed.strip_prefix(CALL_PREFIX) else {
        return malformed("missing `call:` prefix in tool-call payload");
    };
    let Some(open_brace) = rest.find('{') else {
        return malformed("tool-call payload missing `{` after function name");
    };
    let name = rest[..open_brace].trim();
    if name.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::MissingField("name"),
        });
    }
    if !is_valid_function_name(name) {
        return malformed(format!(
            "invalid function name `{name}` — must match [A-Za-z_][A-Za-z0-9_\\-\\.]*"
        ));
    }
    // Trim trailing whitespace on the body before the close-brace
    // check — model output occasionally includes a trailing newline
    // or spaces between `}` and `<tool_call|>` that we tolerate.
    let body = rest[open_brace..].trim_end();
    if !body.ends_with('}') {
        return malformed("tool-call payload missing closing `}`");
    }
    let inner = &body[1..body.len() - 1];
    let args = match parse_object_body(inner) {
        Ok(value) => value,
        Err(message) => return malformed(message),
    };

    // Defensive: the parser should have consumed every paired
    // `<|"|>` quote sentinel. If one survives in the parsed args
    // (anywhere in the JSON) something went wrong — most likely the
    // upstream detokenization stripped one half of a pair, or the
    // model emitted a malformed string. Surface as a hard error
    // rather than ship a bogus tool call to the executor.
    if args_contain_literal_quote_sentinel(&args) {
        return malformed("parsed args still contain a literal `<|\"|>` sentinel");
    }

    if directory.lookup(name).is_none() {
        return PayloadOutcome::Fatal(DecodeEvent::UnknownTool {
            name: name.to_string(),
            raw_args: args,
        });
    }
    let errors = directory.validate_args(name, &args);
    if !errors.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::InvalidArgs {
            name: name.to_string(),
            args,
            errors,
        });
    }

    let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
    PayloadOutcome::Call(vec![
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

/// Function names match `[A-Za-z_][A-Za-z0-9_\-\.]*`. vLLM uses
/// `[\w\-\.]+` and llama.cpp's PEG accepts the same superset; tools in
/// the wild (e.g. `tools.shell-exec`, `web_search.fetch`) need `.` and
/// `-`. Restricting the first character to `[A-Za-z_]` prevents
/// ambiguity with leading digits and the empty-name case.
fn is_valid_function_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

/// Walk a parsed JSON value, returning true if any string anywhere in
/// the structure literally contains `<|"|>` — which the parser should
/// have already consumed as a quote sentinel. Survival means a sentinel
/// was malformed upstream (truncated tokenizer output, half-pair
/// stripped during decode, etc.).
fn args_contain_literal_quote_sentinel(value: &JsonValue) -> bool {
    match value {
        JsonValue::String(s) => s.contains(STRING_QUOTE),
        JsonValue::Array(items) => items.iter().any(args_contain_literal_quote_sentinel),
        JsonValue::Object(map) => map.values().any(args_contain_literal_quote_sentinel),
        _ => false,
    }
}

fn malformed(message: impl Into<String>) -> PayloadOutcome {
    PayloadOutcome::Fatal(DecodeEvent::ParseError {
        sentinel: TOOL_CALL_OPEN,
        source: ParserError::Malformed(message.into()),
    })
}

/// Parse the contents between the outer `{...}` (i.e. without the
/// braces themselves) into a JSON object. Empty input yields an empty
/// object — the model may emit `{}` for parameter-less tools.
fn parse_object_body(body: &str) -> Result<JsonValue, String> {
    let body = body.trim();
    let mut object = JsonMap::new();
    if body.is_empty() {
        return Ok(JsonValue::Object(object));
    }
    for entry in split_top_level(body, ',')? {
        let (key, value) = split_key_value(entry)?;
        object.insert(key, parse_value(value)?);
    }
    Ok(JsonValue::Object(object))
}

fn split_key_value(entry: &str) -> Result<(String, &str), String> {
    let Some(colon) = find_top_level_colon(entry)? else {
        return Err(format!("missing `:` in argument entry `{entry}`"));
    };
    let key = entry[..colon].trim().to_string();
    if key.is_empty() {
        return Err("empty argument key".to_string());
    }
    let value = entry[colon + 1..].trim();
    Ok((key, value))
}

fn parse_value(text: &str) -> Result<JsonValue, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("empty argument value".to_string());
    }
    if let Some(rest) = text.strip_prefix(STRING_QUOTE) {
        let Some(end) = rest.find(STRING_QUOTE) else {
            return Err("unterminated string literal".to_string());
        };
        let after = rest[end + STRING_QUOTE.len()..].trim();
        if !after.is_empty() {
            return Err(format!(
                "trailing characters after string literal: `{after}`"
            ));
        }
        return Ok(JsonValue::String(rest[..end].to_string()));
    }
    if let Some(stripped) = text.strip_prefix('{') {
        let inner = stripped
            .strip_suffix('}')
            .ok_or_else(|| "unterminated nested object".to_string())?;
        return parse_object_body(inner);
    }
    if let Some(stripped) = text.strip_prefix('[') {
        let inner = stripped
            .strip_suffix(']')
            .ok_or_else(|| "unterminated array".to_string())?;
        return parse_array_body(inner);
    }
    match text {
        "true" => Ok(JsonValue::Bool(true)),
        "false" => Ok(JsonValue::Bool(false)),
        "null" | "None" => Ok(JsonValue::Null),
        _ => parse_number(text),
    }
}

fn parse_array_body(body: &str) -> Result<JsonValue, String> {
    let body = body.trim();
    if body.is_empty() {
        return Ok(JsonValue::Array(Vec::new()));
    }
    let mut items = Vec::new();
    for item in split_top_level(body, ',')? {
        items.push(parse_value(item)?);
    }
    Ok(JsonValue::Array(items))
}

fn parse_number(text: &str) -> Result<JsonValue, String> {
    if let Ok(n) = text.parse::<i64>() {
        return Ok(JsonValue::Number(JsonNumber::from(n)));
    }
    if let Ok(n) = text.parse::<u64>() {
        return Ok(JsonValue::Number(JsonNumber::from(n)));
    }
    if let Ok(n) = text.parse::<f64>() {
        if let Some(num) = JsonNumber::from_f64(n) {
            return Ok(JsonValue::Number(num));
        }
    }
    Err(format!("unparseable argument value `{text}`"))
}

/// Split `text` at top-level occurrences of `separator`, respecting:
/// - balanced `{}` and `[]`
/// - paired `<|"|>` delimited strings (treated as opaque)
fn split_top_level(text: &str, separator: char) -> Result<Vec<&str>, String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    let mut in_string = false;

    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if in_string {
            if text[i..].starts_with(STRING_QUOTE) {
                in_string = false;
                i += STRING_QUOTE.len();
                continue;
            }
            i += 1;
            continue;
        }
        if text[i..].starts_with(STRING_QUOTE) {
            in_string = true;
            i += STRING_QUOTE.len();
            continue;
        }
        let ch = bytes[i] as char;
        match ch {
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            c if c == separator && depth_brace == 0 && depth_bracket == 0 => {
                let part = text[start..i].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                start = i + ch.len_utf8();
                i = start;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    if in_string || depth_brace != 0 || depth_bracket != 0 {
        return Err(format!("unterminated tool-call expression: `{text}`"));
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    Ok(parts)
}

fn find_top_level_colon(text: &str) -> Result<Option<usize>, String> {
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    let mut in_string = false;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if in_string {
            if text[i..].starts_with(STRING_QUOTE) {
                in_string = false;
                i += STRING_QUOTE.len();
                continue;
            }
            i += 1;
            continue;
        }
        if text[i..].starts_with(STRING_QUOTE) {
            in_string = true;
            i += STRING_QUOTE.len();
            continue;
        }
        let ch = bytes[i] as char;
        match ch {
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            ':' if depth_brace == 0 && depth_bracket == 0 => return Ok(Some(i)),
            _ => {}
        }
        i += 1;
    }
    if in_string || depth_brace != 0 || depth_bracket != 0 {
        return Err(format!("unterminated tool-call key/value: `{text}`"));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;

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
    fn plain_text_passes_through() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_call_emits_start_args_end() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
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
    fn unknown_tool_is_terminal() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
            &[r#"<|tool_call>call:delete_db{}<tool_call|>"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
            &[r#"<|tool_call>call:calculator{lhs:<|"|>one<|"|>,op:<|"|>add<|"|>,rhs:2}<tool_call|>"#],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "calculator");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_call_prefix_is_terminal() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
            &[r#"<|tool_call>calculator{lhs:1}<tool_call|>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_block_is_terminal() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
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
        let mut p = Gemma4Parser::new(directory());
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(dir);
        let events = run(&mut p, &[r#"<|tool_call>call:ping{}<tool_call|>"#]);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_text_resembling_call_without_sentinel_is_plain_text() {
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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

    #[test]
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = Gemma4Parser::new(directory());
        let first = p.feed(r#"<|tool_call>call:nope{}<tool_call|>"#);
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        let after = p.feed(r#"<|tool_call>call:calculator{lhs:1,op:<|"|>add<|"|>,rhs:2}<tool_call|>"#);
        assert!(after.is_empty());
        let after_finish = p.finish(StopReason::EndOfText);
        assert!(after_finish.is_empty());
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(search_directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(dir);
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
        let mut a = Gemma4Parser::new(dir.clone());
        let mut b = Gemma4Parser::new(dir);
        a.feed(r#"<|tool_call>call:nope{}<tool_call|>"#); // poisons a
        // b must still accept a valid call cleanly.
        let events_b = run(
            &mut b,
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
        let mut p = Gemma4Parser::new(directory());
        let events = run(
            &mut p,
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
