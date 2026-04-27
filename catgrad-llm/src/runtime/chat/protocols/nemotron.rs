//! NVIDIA Nemotron / Nemotron-H tool-call protocol.
//!
//! # Wire format research
//!
//! The reference chat template
//! (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/chat_template.jinja`) renders
//! tool calls as a Llama-3-style XML payload, *not* the Hermes-style
//! JSON of Qwen3:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY1>
//! VALUE1
//! </parameter>
//! <parameter=KEY2>
//! VALUE2
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Each `<tool_call>...</tool_call>` block carries exactly one
//! `<function=...>...</function>` invocation; multiple parallel calls
//! arrive as multiple back-to-back `<tool_call>` blocks (the chat
//! template loops over `message.tool_calls` and emits one block per
//! call).
//!
//! Per-parameter values are stringified via Jinja's `string` filter for
//! scalars and `tojson` for mappings/sequences. To round-trip, this
//! parser tries `serde_json::from_str` on the trimmed value first and
//! falls back to a JSON string on failure — matching the historical
//! `parse_qwen3_5_tool_calls` reference in
//! `catgrad-llm/src/helpers/tool_calls.rs` (now removed; see git
//! history at `675fd07~1`).
//!
//! # Why standalone, not a Qwen3 delegate
//!
//! The outer sentinels `<tool_call>` / `</tool_call>` happen to be
//! identical to Qwen3 — but the payload between them is XML, not
//! Hermes JSON. Delegating to `qwen3::make_parser` would treat the XML
//! body as malformed JSON and fatal-out every call. This module
//! therefore reimplements the state machine (Outside / Inside /
//! Terminated) with an XML payload parser inside the Inside branch.
//!
//! The protocol-registry comment in
//! `runtime/chat/protocol.rs:120` ("Hermes-style JSON in
//! `<tool_call>`") refers to an earlier expectation of the architecture
//! and is inaccurate for the production Nemotron-3-Nano chat template.
//! The registry only stores function pointers, so the wrong comment is
//! cosmetic.
//!
//! # Per-call atomic emission
//!
//! As with the Qwen3 parser, `<tool_call>` opens a buffering mode and
//! the `Start` / `ArgsDelta` / `End` triple is emitted atomically when
//! the matching `</tool_call>` arrives. True per-token argument
//! streaming is out of scope.
//!
//! # Strict gating
//!
//! Bare `<function=...>` syntax outside a `<tool_call>` wrapper is
//! plain text, not a tool call — same rule as the Qwen3 parser's
//! treatment of bare JSON.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

const FUNCTION_OPEN_PREFIX: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN_PREFIX: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// Maximum bytes buffered between `<tool_call>` and `</tool_call>`
/// before the parser fails the call as oversized. Same rationale as
/// the Qwen3 parser: large enough for any plausible structured call,
/// small enough to bound a runaway generation.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Nemotron parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(NemotronParser::new(directory))
}

/// Render the bound tool list into the JSON shape the Nemotron chat
/// template expects.
///
/// The template iterates over `tools` and reads either
/// `tool.function.{name,description,parameters}` or — when the entry is
/// already flat — `tool.{name,description,parameters}`. Emitting the
/// OpenAI envelope `{"type":"function","function":{...}}` keeps the
/// Rust-rendered prompt byte-identical to what
/// `transformers.apply_chat_template` produces from a Python `tools=`
/// argument and matches every other parser in this directory.
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

struct NemotronParser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for `<tool_call>`.
    Outside { matcher: SentinelMatcher },
    /// Inside a tool-call block. Watching for `</tool_call>`; the
    /// matcher's internal buffer is the call's XML payload.
    Inside { matcher: SentinelMatcher },
    /// A fatal protocol error has been emitted. `feed`/`finish` return
    /// empty from this point.
    Terminated,
}

impl NemotronParser {
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

impl IncrementalToolCallParser for NemotronParser {
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

impl NemotronParser {
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
    /// Validated call — emit `Start`, `ArgsDelta`, `End` contiguously.
    Call(Vec<DecodeEvent>),
    /// Anything that should not become a call: unknown name,
    /// schema-invalid args, or a parse failure. Caller wraps with
    /// `Stop { ProtocolError }` and terminates.
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

    // Parse `<function=NAME>...</function>` exactly once.
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

/// Extract `(name, arguments)` from the XML body of one
/// `<tool_call>...</tool_call>` block. The body is expected to contain
/// exactly one `<function=NAME>...</function>` block, optionally with
/// surrounding whitespace.
fn parse_function_block(body: &str) -> Result<(String, JsonValue), ParserError> {
    let (name, function_body) = extract_named_block(body, FUNCTION_OPEN_PREFIX, FUNCTION_CLOSE)?
        .ok_or_else(|| {
            ParserError::Malformed("missing <function=...>...</function> in tool call".into())
        })?;
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(ParserError::MissingField("name"));
    }

    // Inside the function body: zero or more `<parameter=KEY>...</parameter>`
    // blocks, each optionally separated by whitespace.
    let mut arguments = JsonMap::new();
    let mut rest = function_body;
    while let Some(start) = rest.find(PARAMETER_OPEN_PREFIX) {
        let head = &rest[start..];
        let (key, value, consumed) =
            extract_named_block_at_start(head, PARAMETER_OPEN_PREFIX, PARAMETER_CLOSE)?
                .ok_or_else(|| {
                    ParserError::Malformed(
                        "<parameter=...> block missing closing </parameter>".into(),
                    )
                })?;
        let key = key.trim().to_string();
        if key.is_empty() {
            return Err(ParserError::Malformed(
                "<parameter=> block has empty parameter name".into(),
            ));
        }
        // Trim surrounding whitespace; the chat template renders
        // `<parameter=K>\nVALUE\n</parameter>`, so the value contains
        // synthetic newlines we need to drop before reparsing.
        arguments.insert(key, parse_scalar(value.trim()));
        rest = &head[consumed..];
    }

    Ok((name, JsonValue::Object(arguments)))
}

/// Best-effort scalar parse: try JSON first (numbers, booleans, null,
/// nested arrays/objects), fall back to a JSON string literal. Matches
/// the historical `parse_scalar` in `helpers/tool_calls.rs`.
fn parse_scalar(text: &str) -> JsonValue {
    match serde_json::from_str::<JsonValue>(text) {
        Ok(value) => value,
        Err(_) => JsonValue::String(text.to_string()),
    }
}

/// Find a `<prefix NAME>BODY</close>` block anywhere in `text`, return
/// `(name, body)`. Returns `Ok(None)` only if `prefix` is absent
/// entirely; an open-prefix without a matching close is a `Malformed`
/// error.
fn extract_named_block<'a>(
    text: &'a str,
    open_prefix: &str,
    close_tag: &str,
) -> Result<Option<(&'a str, &'a str)>, ParserError> {
    let Some(start) = text.find(open_prefix) else {
        return Ok(None);
    };
    let Some((name, body, _)) =
        extract_named_block_at_start(&text[start..], open_prefix, close_tag)?
    else {
        return Ok(None);
    };
    Ok(Some((name, body)))
}

/// Same as [`extract_named_block`] but requires the open prefix to be
/// at offset 0. Also returns the byte length consumed (open tag through
/// close tag inclusive) so callers can advance past the block.
fn extract_named_block_at_start<'a>(
    text: &'a str,
    open_prefix: &str,
    close_tag: &str,
) -> Result<Option<(&'a str, &'a str, usize)>, ParserError> {
    if !text.starts_with(open_prefix) {
        return Ok(None);
    }
    let header = &text[open_prefix.len()..];
    let Some(name_end) = header.find('>') else {
        return Err(ParserError::Malformed(format!(
            "unterminated `{open_prefix}...>` tag"
        )));
    };
    let name = &header[..name_end];
    let body = &header[name_end + 1..];
    let Some(body_end) = body.find(close_tag) else {
        return Err(ParserError::Malformed(format!(
            "missing `{close_tag}` after `{open_prefix}{name}>`"
        )));
    };
    let consumed = open_prefix.len() + name_end + 1 + body_end + close_tag.len();
    Ok(Some((name, &body[..body_end], consumed)))
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

    /// Render a Nemotron tool-call block. Mirrors the chat template
    /// shape, including the synthetic newlines around values.
    fn call_block(name: &str, args: &[(&str, &str)]) -> String {
        let mut out = String::from("<tool_call>\n<function=");
        out.push_str(name);
        out.push_str(">\n");
        for (k, v) in args {
            out.push_str("<parameter=");
            out.push_str(k);
            out.push_str(">\n");
            out.push_str(v);
            out.push_str("\n</parameter>\n");
        }
        out.push_str("</function>\n</tool_call>");
        out
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
        let mut p = NemotronParser::new(directory_with_add());
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_call_emits_start_args_end() {
        let mut p = NemotronParser::new(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let events = run(&mut p, &[&block]);
        // Start, ArgsDelta, End, Stop
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { index: 0, .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {:?}", events[2]);
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn multiple_calls_in_sequence() {
        let mul = ToolSpec::new(
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
        );
        let dir = Arc::new(ToolDirectory::new(vec![add_tool(), mul]).unwrap());
        let mut p = NemotronParser::new(dir);
        let first = call_block("add", &[("a", "1"), ("b", "2")]);
        let second = call_block("mul", &[("a", "3"), ("b", "4")]);
        let events = run(&mut p, &[&first, &second]);
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
    fn text_then_call_then_text_in_single_feed() {
        let mut p = NemotronParser::new(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let combined = format!("first {block} last");
        let events = run(&mut p, &[&combined]);
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
        let mut p = NemotronParser::new(directory_with_add());
        let block = call_block("delete_db", &[]);
        let events = run(&mut p, &[&block]);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal_with_protocol_error() {
        let mut p = NemotronParser::new(directory_with_add());
        // `a` is given as a string-without-quotes, which `parse_scalar`
        // falls back to a JSON string for — fails the `number` schema.
        let block = call_block("add", &[("a", "one"), ("b", "2")]);
        let events = run(&mut p, &[&block]);
        assert_eq!(events.len(), 2);
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_is_terminal_with_protocol_error() {
        // Open `<function=` with no `>` — unterminated header.
        let mut p = NemotronParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<tool_call><function=add\nno close angle bracket</tool_call>"],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_name_field_is_terminal_with_protocol_error() {
        // `<function=>` — empty name.
        let mut p = NemotronParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<tool_call><function=></function></tool_call>"],
        );
        assert_eq!(events.len(), 2);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::MissingField("name")));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn parameters_key_is_accepted_as_arguments() {
        // The Nemotron wire format does not use a JSON envelope with
        // `arguments` / `parameters` keys (each parameter is its own
        // XML block). The test name is part of the contract checklist;
        // for this protocol the analogous concern is that
        // `<parameter=KEY>` *is* the way to carry arguments — and that
        // we read both numeric and unquoted-string scalar forms back
        // into a plain JSON object.
        let mut p = NemotronParser::new(directory_with_add());
        let block = call_block("add", &[("a", "1"), ("b", "2")]);
        let events = run(&mut p, &[&block]);
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("expected ToolCallEnd, got {events:?}");
        };
        // Arguments arrive as a plain JSON object keyed by parameter name.
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: a JSON-shaped payload not wrapped in `<tool_call>`
        // is plain text — same strict-gating rule as Qwen3.
        let mut p = NemotronParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let mut p = NemotronParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        // No call can have committed yet; no partial-sentinel bytes
        // should leak as text.
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains('<'),
            _ => true,
        }));
        events.extend(p.feed(
            "all>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> done",
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
        let mut p = NemotronParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_",
        ));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let mut p = NemotronParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<tool_call>\n<function=add>\n<parameter=a>\n1"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = NemotronParser::new(directory_with_add());
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
            DecodeEvent::Stop { reason: StopReason::ProtocolError }
        ));
    }

    #[test]
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = NemotronParser::new(directory_with_add());
        let block = call_block("delete_db", &[]);
        let first = p.feed(&block);
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &first[1],
            DecodeEvent::Stop { reason: StopReason::ProtocolError }
        ));
        let after_feed = p.feed("any further text");
        assert!(after_feed.is_empty(), "got events: {after_feed:?}");
        let after_more = p.feed(&call_block("add", &[("a", "1"), ("b", "2")]));
        assert!(after_more.is_empty(), "got events: {after_more:?}");
        let after_finish = p.finish(StopReason::EndOfText);
        assert!(after_finish.is_empty(), "got events: {after_finish:?}");
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

    // --- payload-shape unit tests (do not exercise the state machine) ---

    #[test]
    fn parse_scalar_handles_json_and_strings() {
        assert_eq!(parse_scalar("42"), json!(42));
        assert_eq!(parse_scalar("3.14"), json!(3.14));
        assert_eq!(parse_scalar("true"), json!(true));
        assert_eq!(parse_scalar("null"), json!(null));
        assert_eq!(parse_scalar("[1, 2]"), json!([1, 2]));
        assert_eq!(parse_scalar("\"quoted\""), json!("quoted"));
        // Unquoted text falls back to a string.
        assert_eq!(parse_scalar("hello"), json!("hello"));
    }

    #[test]
    fn parse_function_block_extracts_parameters() {
        let body =
            "<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>";
        let (name, args) = parse_function_block(body).unwrap();
        assert_eq!(name, "add");
        assert_eq!(args, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn parse_function_block_no_parameters_yields_empty_args() {
        let body = "<function=do_thing>\n</function>";
        let (name, args) = parse_function_block(body).unwrap();
        assert_eq!(name, "do_thing");
        assert_eq!(args, json!({}));
    }
}

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across chunk boundaries produces the same final
    //! decoded turn (or the same DecodeFailure).

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // single valid call (matching the `add` tool used by the
            // shared test directory in `test_util`)
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>",
            // call surrounded by text
            "prefix <tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> suffix",
            // two calls back-to-back
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=add>\n<parameter=a>\n3\n</parameter>\n<parameter=b>\n4\n</parameter>\n</function>\n</tool_call>",
            // sentinel-shaped text that isn't a sentinel
            "the docs say <tool_call> but it's just text",
            // unknown tool — fatal, but chunk-invariance still holds
            "<tool_call>\n<function=missing>\n</function>\n</tool_call>",
            // call with text suffix only
            "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call> done",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..7,
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
            input_idx in 0_usize..7,
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
