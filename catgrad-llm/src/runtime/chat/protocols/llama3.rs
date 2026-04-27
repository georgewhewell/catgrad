//! Llama 3.x tool-call protocol (Llama 3.1 / 3.2 / 3.3 / 4 Instruct).
//!
//! # Wire format
//!
//! Unlike the Hermes / Qwen3 family, Llama 3.x emits tool calls as
//! **bare JSON objects** with no enclosing sentinels — the entire
//! assistant message is the JSON, optionally prefixed by the
//! `<|python_tag|>` special token:
//!
//! ```text
//! {"name": "calculator", "parameters": {"lhs": 7, "rhs": 5, "op": "add"}}
//! ```
//!
//! or
//!
//! ```text
//! <|python_tag|>{"name": "calculator", "parameters": {"lhs": 7, "rhs": 5, "op": "add"}}
//! ```
//!
//! The Llama 3 chat template natively renders the tool spec into the
//! system / first-user message and instructs the model to "respond in
//! the format `{"name": ..., "parameters": ...}`." Stop is the
//! `<|eot_id|>` token which the runtime catches via `eos_token_ids`.
//!
//! # Why this needs a different parser from Hermes
//!
//! The Hermes-style parsers buffer between paired sentinels; here the
//! "open sentinel" is implicit (a leading `{` after optional whitespace
//! or `<|python_tag|>`) and the "close sentinel" is the end of the
//! generation itself. So the parser cannot commit a tool call until
//! `finish` is called — there is no in-stream signal that the JSON
//! body is complete.
//!
//! ## Streaming behaviour
//!
//! - **First non-whitespace char is `{`** (or `<|python_tag|>{`):
//!   buffer everything in `feed`, parse and emit the call atomically
//!   on `finish`. Streaming of partial JSON is not supported (would
//!   require a partial-JSON parser and validation rollback).
//! - **First non-whitespace char is anything else:** stream as text,
//!   passthrough on every `feed`. This preserves token-by-token
//!   streaming for plain-text replies.
//!
//! If the buffered JSON fails to parse, or decodes to something other
//! than a `{name, parameters}` object, the parser falls back to
//! emitting the raw buffer as a single `TextDelta`. This is more
//! forgiving than the strict Hermes parser because the model has no
//! way to signal "this is just text" inside the bare-JSON dialect —
//! treating malformed tool-call attempts as text gives the user
//! something rather than a fatal protocol error.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, StopReason, ToolDirectory, ToolSpec,
};
use crate::types;

use crate::runtime::chat::codecs::json::peel_spec_shape_echo;

/// Optional prefix that the Llama 3 reasoning models emit before the
/// JSON body. We strip it before deciding whether the body is a tool
/// call — the model uses it primarily for the built-in `ipython`
/// environment, but can leak it into custom-tool replies when the
/// system message advertises ipython tools.
const PYTHON_TAG: &str = "<|python_tag|>";

/// Hard cap on the bytes buffered while we wait for the end of a
/// candidate JSON tool-call. Mirrors the Hermes parser's limit so a
/// runaway generation cannot exhaust gateway memory.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Llama-3 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Llama3Parser::new(directory))
}


/// Llama 3's chat template renders tool definitions itself when the
/// `tools` variable is bound. Identity over the message list.
pub fn prepare_messages(_specs: &[ToolSpec], messages: Vec<types::Message>) -> Vec<types::Message> {
    messages
}

struct Llama3Parser {
    directory: Arc<ToolDirectory>,
    state: State,
}

enum State {
    /// Skipping leading whitespace and `<|python_tag|>`. We buffer the
    /// whitespace / tag itself in case the body turns out to be plain
    /// text (so we don't lose the leading bytes).
    LeadingTrivia { buffer: String },
    /// Saw `{` (or `<|python_tag|>{`) — accumulating the candidate
    /// JSON body. At `finish`, we'll try to parse and emit either a
    /// tool call or a TextDelta.
    Buffering { buffer: String },
    /// First non-whitespace char wasn't `{` — pass everything through
    /// as TextDelta. The matched-prefix string is already emitted.
    Streaming,
    /// A fatal protocol error has been emitted.
    Terminated,
}

impl Llama3Parser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self {
            directory,
            state: State::LeadingTrivia {
                buffer: String::new(),
            },
        }
    }
}

impl IncrementalToolCallParser for Llama3Parser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        // Each iteration consumes `remaining` exactly once into the
        // current state's buffer, then either transitions or breaks.
        // We use `Option::take` to make the "consumed" lifetime
        // explicit; subsequent iterations operate on already-buffered
        // data (no re-injection of `remaining`).
        let mut remaining = Some(text);

        loop {
            match &mut self.state {
                State::LeadingTrivia { buffer } => {
                    if let Some(chunk) = remaining.take() {
                        buffer.push_str(chunk);
                    }
                    match classify_leading_trivia(buffer) {
                        LeadingClassification::NeedMore => break,
                        LeadingClassification::ToolCall { body_start } => {
                            // Drop the trivia (whitespace / python_tag);
                            // start buffering the JSON body.
                            let body = buffer[body_start..].to_string();
                            self.state = State::Buffering { buffer: body };
                        }
                        LeadingClassification::Text => {
                            // Trivia turned out to be just plain text.
                            // Drain what we have as a TextDelta and
                            // switch to streaming passthrough.
                            let drained = std::mem::take(buffer);
                            if !drained.is_empty() {
                                events.push(DecodeEvent::TextDelta(drained));
                            }
                            self.state = State::Streaming;
                        }
                    }
                }
                State::Buffering { buffer } => {
                    if let Some(chunk) = remaining.take() {
                        buffer.push_str(chunk);
                    }
                    if buffer.len() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: PYTHON_TAG,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    break;
                }
                State::Streaming => {
                    if let Some(chunk) = remaining.take() {
                        if !chunk.is_empty() {
                            events.push(DecodeEvent::TextDelta(chunk.to_string()));
                        }
                    }
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
        let state = std::mem::replace(&mut self.state, State::Terminated);
        match state {
            State::LeadingTrivia { buffer } => {
                if !buffer.is_empty() {
                    events.push(DecodeEvent::TextDelta(buffer));
                }
            }
            State::Streaming => {}
            State::Buffering { buffer } => {
                events.extend(finalize_buffered_call(&buffer, &self.directory));
            }
            State::Terminated => unreachable!("checked above"),
        }
        // Fatal events (UnknownTool / InvalidArgs / ParseError) demand
        // `StopReason::ProtocolError`; any successful path uses the
        // caller's reason. This keeps `finalize_buffered_call` free
        // of the Stop concern — it just produces the call/error event.
        let stop_reason = if events.iter().any(is_fatal_event) {
            StopReason::ProtocolError
        } else {
            reason
        };
        events.push(DecodeEvent::Stop { reason: stop_reason });
        events
    }
}

impl Llama3Parser {
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

fn is_fatal_event(ev: &DecodeEvent) -> bool {
    matches!(
        ev,
        DecodeEvent::UnknownTool { .. }
            | DecodeEvent::InvalidArgs { .. }
            | DecodeEvent::ParseError { .. }
    )
}

enum LeadingClassification {
    /// Need more bytes to disambiguate (e.g. partial `<|python_tag|>`
    /// arriving across feeds).
    NeedMore,
    /// Saw `{` after optional whitespace and `<|python_tag|>`. Body
    /// starts at byte offset `body_start` of the trivia buffer.
    ToolCall { body_start: usize },
    /// First non-whitespace, non-`<|python_tag|>` byte wasn't `{`.
    Text,
}

fn classify_leading_trivia(buffer: &str) -> LeadingClassification {
    // Skip whitespace and an optional <|python_tag|>.
    let after_ws = buffer.trim_start();
    if after_ws.is_empty() {
        // Pure whitespace so far — keep waiting.
        return LeadingClassification::NeedMore;
    }
    let (after_python_tag, _ate_python_tag) = if let Some(rest) = after_ws.strip_prefix(PYTHON_TAG)
    {
        (rest, true)
    } else if PYTHON_TAG.starts_with(after_ws) {
        // Buffer is a strict prefix of `<|python_tag|>` (e.g. `<|py`).
        // Could resolve either way; need more bytes.
        return LeadingClassification::NeedMore;
    } else {
        (after_ws, false)
    };

    // Compute byte offset of `after_python_tag` within `buffer` so the
    // caller can resume Buffering at exactly the JSON body.
    //
    // SAFETY: both `after_python_tag` and `buffer` share an allocation
    // because `after_python_tag` comes from `buffer.trim_start()` and
    // `strip_prefix` (each returns a sub-slice of `buffer`). Using the
    // pointer arithmetic here is the standard pattern for recovering an
    // offset from a sub-slice.
    let body_start = (after_python_tag.as_ptr() as usize) - (buffer.as_ptr() as usize);

    // Body bytes haven't arrived yet — e.g. the buffer ends with the
    // python_tag exactly, or with "<|python_tag|> " (whitespace
    // already trimmed in `after_ws`, but `strip_prefix` may have left
    // nothing behind on a chunk boundary).
    let Some(first_non_ws) = after_python_tag.chars().next() else {
        return LeadingClassification::NeedMore;
    };
    if first_non_ws == '{' {
        LeadingClassification::ToolCall { body_start }
    } else {
        LeadingClassification::Text
    }
}

fn finalize_buffered_call(buffer: &str, directory: &ToolDirectory) -> Vec<DecodeEvent> {
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    // Llama 3 sometimes emits multiple tool calls back-to-back as
    // `{...}{...}` (no separator), or `{...}\n{...}`, or with `;`
    // separators per Meta's spec examples. Use a streaming
    // deserializer to consume each top-level value and stop on the
    // first parse error. This is the same pattern vLLM's Llama parser
    // uses (`raw_decode` end-index loop).
    let mut values = Vec::new();
    let mut stream = serde_json::Deserializer::from_str(trimmed).into_iter::<JsonValue>();
    while let Some(next) = stream.next() {
        match next {
            Ok(v) => values.push(v),
            Err(_) => break,
        }
    }

    if values.is_empty() {
        // No valid JSON at all — fall back to plain text. Bare-JSON
        // dialect has no escape for "I meant this as content."
        return vec![DecodeEvent::TextDelta(buffer.to_string())];
    }

    let mut events = Vec::with_capacity(values.len() * 3);
    for (index, value) in values.into_iter().enumerate() {
        match build_call_events(index, value, directory) {
            Ok(triple) => events.extend(triple),
            Err(fatal) => {
                // First fatal stops the loop and short-circuits to the
                // error path. Llama's chat template only allows one
                // call per turn anyway, so this matches the model
                // contract.
                return if events.is_empty() {
                    // The first value was already invalid: render the
                    // structured error or fall back to text.
                    fatal_or_text(fatal, buffer)
                } else {
                    // We already validated some calls; surface the
                    // fatal as the terminal event so the gateway sees
                    // both the partial success and the failure.
                    let mut out = events;
                    out.push(fatal);
                    out
                };
            }
        }
    }
    events
}

/// If the fatal event is a "structural" error (UnknownTool /
/// InvalidArgs) emit it. If it's just "this isn't a call shape after
/// all" (Malformed / MissingField), emit the buffer as text — the
/// bare-JSON dialect can't distinguish "the model meant this as text"
/// from "the model emitted invalid call JSON."
fn fatal_or_text(fatal: DecodeEvent, buffer: &str) -> Vec<DecodeEvent> {
    match &fatal {
        DecodeEvent::UnknownTool { .. } | DecodeEvent::InvalidArgs { .. } => vec![fatal],
        _ => vec![DecodeEvent::TextDelta(buffer.to_string())],
    }
}

fn build_call_events(
    index: usize,
    value: JsonValue,
    directory: &ToolDirectory,
) -> Result<Vec<DecodeEvent>, DecodeEvent> {
    let object = peel_spec_shape_echo(value).ok_or_else(|| {
        DecodeEvent::ParseError {
            sentinel: PYTHON_TAG,
            source: ParserError::Malformed("not a JSON object".into()),
        }
    })?;
    let Some(name) = object.get("name").and_then(JsonValue::as_str) else {
        return Err(DecodeEvent::ParseError {
            sentinel: PYTHON_TAG,
            source: ParserError::MissingField("name"),
        });
    };
    // Llama 3 spec: `parameters`. Tolerate `arguments`; vLLM's Llama
    // parser actually prefers `arguments` if both are present.
    let args = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .cloned()
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));

    if directory.lookup(name).is_none() {
        return Err(DecodeEvent::UnknownTool {
            name: name.to_string(),
            raw_args: args,
        });
    }
    let errors = directory.validate_args(name, &args);
    if !errors.is_empty() {
        return Err(DecodeEvent::InvalidArgs {
            name: name.to_string(),
            args,
            errors,
        });
    }

    let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
    Ok(vec![
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::protocol_test_kit::{
        add_tool, directory_with_add, last_stop_reason, run,
    };
    use crate::runtime::chat::{
        DecodeEvent, IncrementalToolCallParser, StopReason, ToolDirectory, ToolSpec,
    };
    use serde_json::json;

    #[test]
    fn plain_text_streams_through() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["hello ", "world"]);
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "hello world");
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::ToolCallStart { .. })));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn bare_json_emits_tool_call() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"name":"add","parameters":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn bare_json_with_python_tag_emits_tool_call() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<|python_tag|>{"name":"add","parameters":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    #[test]
    fn arguments_key_is_accepted_alongside_parameters() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    #[test]
    fn leading_whitespace_does_not_prevent_tool_detection() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &["   ", "\n", r#"{"name":"add","parameters":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
    }

    #[test]
    fn partial_python_tag_split_across_feeds() {
        let mut p = make_parser(directory_with_add());
        let events_partial = p.feed("<|py");
        // Cannot have committed yet — `<|py` is a strict prefix of
        // `<|python_tag|>`, so we hold off.
        assert!(events_partial.is_empty(), "events: {events_partial:?}");
        let events_rest = p.feed(r#"thon_tag|>{"name":"add","parameters":{"a":1,"b":2}}"#);
        let mut events = events_partial;
        events.extend(events_rest);
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
    }

    #[test]
    fn malformed_json_falls_back_to_text_delta() {
        // Bare-JSON dialect has no "this is just text" escape, so a
        // syntactically-broken JSON should not be a fatal error — we
        // emit it as text and let the user see the raw output.
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &[r#"{"name":"add","parameters":{"a":1"#]);
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert!(
            !text.is_empty(),
            "expected text fallback for unfinished JSON; got {events:?}"
        );
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unknown_tool_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &[r#"{"name":"delete_db","parameters":{}}"#]);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal_with_protocol_error() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"name":"add","parameters":{"a":"oops"}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::InvalidArgs { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = make_parser(directory_with_add());
        let mut chunk = String::from("{");
        chunk.push_str(&"x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1));
        let events = p.feed(&chunk);
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::PayloadTooLarge { .. },
                ..
            }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    /// Spec-shape echo: small Llama Instruct fine-tunes (notably
    /// Llama-3.2-1B) often regurgitate the OpenAI tool-spec envelope
    /// the chat template handed them, instead of Meta's documented
    /// response shape. Peel and recover.
    #[test]
    fn parser_peels_openai_function_envelope_with_inner_object() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"type":"function","function":{"name":"add","arguments":{"a":1,"b":2}}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    /// Variant the user actually saw on Llama-3.2-1B-Instruct:
    /// `function` is the tool name as a string, args under outer
    /// `parameters`. This is malformed by every spec but the model
    /// emits it.
    #[test]
    fn parser_recovers_function_string_with_outer_parameters() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"type":"function","function":"add","parameters":{"a":1,"b":2}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    /// Llama 3 chat template raises on multiple `tool_calls`, so the
    /// dialect is structurally single-call. But models occasionally
    /// emit two JSON objects back-to-back. The parser surfaces both
    /// as separate calls rather than silently dropping the second.
    #[test]
    fn parser_accepts_multiple_back_to_back_calls() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"{"name":"add","parameters":{"a":1,"b":2}}{"name":"add","parameters":{"a":3,"b":4}}"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { index: 0, .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallStart { index: 1, .. }));
    }

    #[test]
    fn pure_whitespace_with_no_body_emits_text_only() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["   "]);
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "   ");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }
}

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across chunk boundaries produces the same final
    //! decoded turn.

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // bare JSON tool call
            r#"{"name":"add","parameters":{"a":1,"b":2}}"#,
            // python_tag-prefixed tool call
            r#"<|python_tag|>{"name":"add","parameters":{"a":1,"b":2}}"#,
            // leading whitespace
            r#"   {"name":"add","parameters":{"a":1,"b":2}}"#,
            // text-shaped (does not start with `{`)
            "Sure, the answer is 7+5=12.",
            // unknown tool
            r#"{"name":"missing","parameters":{}}"#,
            // schema-invalid
            r#"{"name":"add","parameters":{"a":"x"}}"#,
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
