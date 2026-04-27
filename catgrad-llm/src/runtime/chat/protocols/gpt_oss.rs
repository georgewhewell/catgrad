//! gpt-oss / OpenAI **harmony** tool-call protocol.
//!
//! Unlike the sentinel-pair dialects (Qwen3, LFM2, etc.), harmony is a
//! token-stream format with its own channel routing. A single
//! generation can interleave multiple "messages", each tagged with a
//! channel (`final`, `commentary`, `analysis`). Tool calls live in the
//! `commentary` channel with a `to=functions.NAME` recipient header.
//! The `final` channel carries the user-visible answer.
//!
//! # Confirmed wire format
//!
//! Special tokens, exact byte strings, from the openai/harmony Rust
//! source (`FormattingToken::as_str()`):
//!
//! ```text
//! <|start|>     (id 200006)  — begin a message header
//! <|end|>       (id 200007)  — terminate a non-final message
//! <|message|>   (id 200008)  — header → body separator
//! <|channel|>   (id 200005)  — header begins channel block
//! <|return|>    (id 200002)  — terminate a final-channel message
//! <|call|>      (id 200012)  — terminate a commentary tool-call message
//! <|constrain|> (id 200003)  — content-type marker (e.g. `json`)
//! ```
//!
//! Reference: <https://developers.openai.com/cookbook/articles/openai-harmony>
//! and <https://github.com/openai/harmony/blob/main/src/encoding.rs>.
//!
//! ## Message shapes
//!
//! - User-visible answer (`final` channel):
//!
//!   ```text
//!   <|start|>assistant<|channel|>final<|message|>{text}<|return|>
//!   ```
//!
//!   `<|return|>` is the model's "I'm done" stop token.
//!
//! - Tool call (`commentary` channel addressed to a function):
//!
//!   ```text
//!   <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{json args}<|call|>
//!   ```
//!
//!   Tokens between `<|channel|>` and `<|message|>` are space-separated;
//!   the channel name is read up to whitespace or `<`, then `to=` and
//!   `<|constrain|>json` may appear in any order before `<|message|>`.
//!
//! - Chain-of-thought (`analysis` channel):
//!
//!   ```text
//!   <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
//!   ```
//!
//! Multiple messages may chain in one generation. The `<|start|>assistant`
//! prefix is required to begin each NEW message; in catgrad's detokenizer
//! stream the very first message of a turn typically arrives without that
//! prefix (since the chat template emitted `<|start|>assistant` as the
//! prompt's last tokens), so we accept either entry point.
//!
//! # Channel routing decisions
//!
//! - `final` body → [`DecodeEvent::TextDelta`]. Streamed as the model
//!   emits it, since the user sees this directly.
//! - `commentary` with `to=functions.NAME` → tool call. Body buffered
//!   until `<|call|>`, then parsed as JSON, validated, and emitted as
//!   the atomic `Start` / `ArgsDelta` / `End` triple.
//! - `commentary` *without* a `to=functions.*` recipient → dropped.
//!   These are the "tool-calling preamble" / "I'm about to do X"
//!   narrations that the harmony spec earmarks for the commentary
//!   channel; the [`DecodeEvent`] surface has no place to put them and
//!   surfacing them as user text would mis-render.
//! - `analysis` → dropped. Reasoning content is private to the model;
//!   the [`DecodedAssistantTurn`](super::super::DecodedAssistantTurn)
//!   surface intentionally does not expose it. Document choice: drop
//!   silently rather than fail, so a `analysis`-then-`final` generation
//!   produces just the user-visible answer.
//! - Unknown channel name → fatal `ParseError`. Defensive: a model
//!   generating a channel other than these three is producing
//!   off-distribution output and should not be silently passed.
//!
//! # Tool-call recipient parsing
//!
//! The harmony header is `commentary to=functions.NAME [...]`. We
//! split by ASCII whitespace, accept tokens that start with `to=`
//! (case-sensitive), strip the `functions.` namespace, and take the
//! tail as the tool name. Tokens that don't match a recognized header
//! shape (e.g. `<|constrain|>json`) are ignored; the harmony parser
//! itself is permissive about token order.
//!
//! # Strict gating
//!
//! Plain text outside any channel emits as `TextDelta`. Real gpt-oss
//! output always wraps content in channel headers — but we handle the
//! defensive case so a buggy detokenizer or a partially-stripped prompt
//! prefix doesn't produce a crash.
//!
//! # Per-call atomic emission
//!
//! Same as the sentinel-pair dialects: a tool call is buffered until
//! `<|call|>` arrives, then validated and emitted as one
//! `Start` / `ArgsDelta` / `End` triple. Multiple commentary messages
//! per generation produce sequential indices (0, 1, ...).
//!
//! # Trait-contract conflicts
//!
//! None blocking, but worth recording: the harmony header carries
//! semantically distinct kinds of data (channel, recipient, content
//! type) that the [`IncrementalToolCallParser`] surface flattens away.
//! The parser absorbs all of that in its state machine and emits only
//! the wire-neutral [`DecodeEvent`] vocabulary the rest of the stack
//! agreed on. A future need to surface analysis-channel reasoning to
//! clients would require a new event variant — out of scope here.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory,
};

// --- Special-token strings (exact byte sequences, see module doc) ---

const TOK_START: &str = "<|start|>";
const TOK_CHANNEL: &str = "<|channel|>";
const TOK_MESSAGE: &str = "<|message|>";
const TOK_END: &str = "<|end|>";
const TOK_RETURN: &str = "<|return|>";
const TOK_CALL: &str = "<|call|>";

/// Sentinel name reported in `ParseError` events. Picked to be the most
/// generally-recognizable harmony token in operator messages.
const ERR_SENTINEL: &str = "<|channel|>";

/// Hard cap on the body of any single channel message. Same rationale
/// as the other dialects: large enough for any plausible tool-call
/// JSON, small enough that a runaway generation cannot exhaust gateway
/// memory. Final-channel text bodies are also subject to this cap, but
/// we flush user-facing text to `TextDelta` events incrementally so the
/// cap effectively only fires on commentary/analysis bodies that are
/// buffered to EOM.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Construct a gpt-oss harmony parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(GptOssParser::new(directory))
}

/// Render the bound tool list into the JSON shape the gpt-oss harmony
/// chat template expects.
///
/// The template iterates `tools` and reads `tool.function.name`,
/// `tool.function.description`, `tool.function.parameters`, then emits
/// a TypeScript-namespace block under `namespace functions`. So the
/// shape is the OpenAI envelope: `{"type":"function","function":{...}}`.
///

// --- Parser state machine ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Final,
    Commentary,
    Analysis,
}

struct GptOssParser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

/// Outer state. Each variant owns the lookahead buffer it needs.
enum State {
    /// Outside any header or body. Watching for either `<|channel|>`
    /// (the bare entry point — most-common since the prompt ended with
    /// `<|start|>assistant`) or `<|start|>` (which begins a fresh
    /// message header inside a multi-message generation).
    ///
    /// The pair-matcher pattern of one matcher per sentinel doesn't
    /// fit directly because we have *two* possible terminators. We
    /// keep an `outside_buf` and search both sentinels by hand.
    Outside { outside_buf: String },
    /// Just saw `<|start|>`. Read the role (`assistant`) up to the next
    /// `<|channel|>` (its ONLY valid follower in this position). The
    /// role is currently always `assistant` in catgrad's gateway flow.
    /// Defensive: any role + `<|channel|>` is accepted.
    InStartHeader { matcher: SentinelMatcher },
    /// Just saw `<|channel|>`. Reading channel name and any header
    /// tokens (`to=...`, `<|constrain|>json`) up to `<|message|>`.
    InChannelHeader { matcher: SentinelMatcher },
    /// Inside a message body. The terminator depends on the channel:
    /// `final` ends on `<|return|>` or `<|end|>`; `commentary` to a
    /// function ends on `<|call|>`; bodies that don't end on their
    /// expected terminator but on a different one are also handled
    /// (see body-loop logic).
    InBody {
        channel: Channel,
        recipient: Option<String>,
        body: String,
    },
    /// Fatal error already emitted. `feed`/`finish` return empty.
    Terminated,
}

impl GptOssParser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self {
            directory,
            state: State::Outside {
                outside_buf: String::new(),
            },
            next_index: 0,
        }
    }
}

impl IncrementalToolCallParser for GptOssParser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        let mut remaining = text.to_string();
        loop {
            // Move the current state out so we can match by ownership;
            // each branch reinstalls (or transitions) its own state.
            let state = std::mem::replace(
                &mut self.state,
                State::Outside {
                    outside_buf: String::new(),
                },
            );
            let progress = match state {
                State::Outside { outside_buf } => {
                    self.step_outside(outside_buf, &mut remaining, &mut events)
                }
                State::InStartHeader { matcher } => {
                    self.step_start_header(matcher, &mut remaining, &mut events)
                }
                State::InChannelHeader { matcher } => {
                    self.step_channel_header(matcher, &mut remaining, &mut events)
                }
                State::InBody {
                    channel,
                    recipient,
                    body,
                } => self.step_body(channel, recipient, body, &mut remaining, &mut events),
                State::Terminated => StepResult::Done,
            };
            match progress {
                StepResult::Continue => continue,
                StepResult::Done => break,
            }
        }
        events
    }

    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let state = std::mem::replace(
            &mut self.state,
            State::Outside {
                outside_buf: String::new(),
            },
        );
        match state {
            State::Outside { outside_buf } => {
                // Anything still in the buffer is plain text — it
                // cannot extend into a sentinel match because we keep
                // outside_buf bounded only by the longest sentinel
                // prefix (handled implicitly: at finish() time we just
                // emit everything).
                let mut events = Vec::new();
                if !outside_buf.is_empty() {
                    events.push(DecodeEvent::TextDelta(outside_buf));
                }
                events.push(DecodeEvent::Stop { reason });
                events
            }
            State::InStartHeader { .. }
            | State::InChannelHeader { .. }
            | State::InBody { .. } => self.fatal(DecodeEvent::ParseError {
                sentinel: ERR_SENTINEL,
                source: ParserError::Unterminated,
            }),
            State::Terminated => unreachable!("checked above"),
        }
    }
}

/// Whether the loop should continue (state changed, more input may be
/// processable) or stop (waiting for more input or fatal-terminated).
enum StepResult {
    Continue,
    Done,
}

impl GptOssParser {
    fn fatal(&mut self, error_event: DecodeEvent) -> Vec<DecodeEvent> {
        self.state = State::Terminated;
        vec![
            error_event,
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError,
            },
        ]
    }

    /// Outside any header. Search for the earliest of `<|start|>` and
    /// `<|channel|>`; emit any preceding plain text and transition to
    /// the corresponding header state.
    fn step_outside(
        &mut self,
        mut outside_buf: String,
        remaining: &mut String,
        events: &mut Vec<DecodeEvent>,
    ) -> StepResult {
        outside_buf.push_str(remaining);
        remaining.clear();

        // Find the earliest match of either sentinel.
        let start_pos = outside_buf.find(TOK_START);
        let channel_pos = outside_buf.find(TOK_CHANNEL);
        match (start_pos, channel_pos) {
            (Some(s), Some(c)) if s <= c => {
                self.commit_start_match(s, &outside_buf, events);
                let after = outside_buf[s + TOK_START.len()..].to_string();
                *remaining = after;
                StepResult::Continue
            }
            (Some(s), None) => {
                self.commit_start_match(s, &outside_buf, events);
                let after = outside_buf[s + TOK_START.len()..].to_string();
                *remaining = after;
                StepResult::Continue
            }
            (Some(_s), Some(c)) => {
                // c < s — channel comes first.
                self.commit_channel_match(c, &outside_buf, events);
                let after = outside_buf[c + TOK_CHANNEL.len()..].to_string();
                *remaining = after;
                StepResult::Continue
            }
            (None, Some(c)) => {
                self.commit_channel_match(c, &outside_buf, events);
                let after = outside_buf[c + TOK_CHANNEL.len()..].to_string();
                *remaining = after;
                StepResult::Continue
            }
            (None, None) => {
                // No sentinel seen. Emit safe text (the part that
                // can't extend into either sentinel) and re-park
                // the rest in outside_buf.
                let safe_end = safe_emit_boundary_for_two(&outside_buf, TOK_START, TOK_CHANNEL);
                let safe = outside_buf[..safe_end].to_string();
                let park = outside_buf[safe_end..].to_string();
                if !safe.is_empty() {
                    events.push(DecodeEvent::TextDelta(safe));
                }
                self.state = State::Outside { outside_buf: park };
                StepResult::Done
            }
        }
    }

    fn commit_start_match(&mut self, pos: usize, outside_buf: &str, events: &mut Vec<DecodeEvent>) {
        let before = &outside_buf[..pos];
        if !before.is_empty() {
            events.push(DecodeEvent::TextDelta(before.to_string()));
        }
        self.state = State::InStartHeader {
            matcher: SentinelMatcher::new(TOK_CHANNEL),
        };
    }

    fn commit_channel_match(
        &mut self,
        pos: usize,
        outside_buf: &str,
        events: &mut Vec<DecodeEvent>,
    ) {
        let before = &outside_buf[..pos];
        if !before.is_empty() {
            events.push(DecodeEvent::TextDelta(before.to_string()));
        }
        self.state = State::InChannelHeader {
            matcher: SentinelMatcher::new(TOK_MESSAGE),
        };
    }

    /// Inside `<|start|>...<|channel|>` — between the start of a new
    /// message and the channel marker. The header carries the role
    /// (typically `assistant`); we don't currently use it. When we hit
    /// `<|channel|>`, transition to channel-header reading.
    fn step_start_header(
        &mut self,
        mut matcher: SentinelMatcher,
        remaining: &mut String,
        events: &mut Vec<DecodeEvent>,
    ) -> StepResult {
        matcher.push(remaining);
        remaining.clear();
        if matcher.buffered_bytes() > MAX_BODY_BYTES {
            events.extend(self.fatal(DecodeEvent::ParseError {
                sentinel: ERR_SENTINEL,
                source: ParserError::PayloadTooLarge {
                    limit_bytes: MAX_BODY_BYTES,
                },
            }));
            return StepResult::Done;
        }
        if let Some((_role, after)) = matcher.try_match() {
            // We don't validate role; harmony's parser is permissive,
            // and assistant is the only role catgrad's gateway flow
            // produces in this position.
            self.state = State::InChannelHeader {
                matcher: SentinelMatcher::new(TOK_MESSAGE),
            };
            *remaining = after;
            StepResult::Continue
        } else {
            self.state = State::InStartHeader { matcher };
            StepResult::Done
        }
    }

    /// Inside a channel header — between `<|channel|>` and `<|message|>`.
    /// Read until `<|message|>`, then parse the header tokens to extract
    /// the channel name and recipient.
    fn step_channel_header(
        &mut self,
        mut matcher: SentinelMatcher,
        remaining: &mut String,
        events: &mut Vec<DecodeEvent>,
    ) -> StepResult {
        matcher.push(remaining);
        remaining.clear();
        if matcher.buffered_bytes() > MAX_BODY_BYTES {
            events.extend(self.fatal(DecodeEvent::ParseError {
                sentinel: ERR_SENTINEL,
                source: ParserError::PayloadTooLarge {
                    limit_bytes: MAX_BODY_BYTES,
                },
            }));
            return StepResult::Done;
        }
        if let Some((header_text, after)) = matcher.try_match() {
            match parse_channel_header(&header_text) {
                Ok((channel, recipient)) => {
                    self.state = State::InBody {
                        channel,
                        recipient,
                        body: String::new(),
                    };
                    *remaining = after;
                    StepResult::Continue
                }
                Err(err) => {
                    events.extend(self.fatal(DecodeEvent::ParseError {
                        sentinel: ERR_SENTINEL,
                        source: err,
                    }));
                    StepResult::Done
                }
            }
        } else {
            self.state = State::InChannelHeader { matcher };
            StepResult::Done
        }
    }

    /// Inside a message body. Three possible terminators: `<|call|>`
    /// (commentary tool call), `<|return|>` (final answer), `<|end|>`
    /// (analysis or non-final). Final-channel text is emitted
    /// incrementally as `TextDelta` so the user sees streaming output;
    /// commentary/analysis bodies are buffered for end-of-message
    /// validation.
    fn step_body(
        &mut self,
        channel: Channel,
        recipient: Option<String>,
        mut body: String,
        remaining: &mut String,
        events: &mut Vec<DecodeEvent>,
    ) -> StepResult {
        body.push_str(remaining);
        remaining.clear();

        if body.len() > MAX_BODY_BYTES {
            // Hard cap. Fires only on commentary/analysis bodies in
            // practice: final-channel text is drained below.
            events.extend(self.fatal(DecodeEvent::ParseError {
                sentinel: ERR_SENTINEL,
                source: ParserError::PayloadTooLarge {
                    limit_bytes: MAX_BODY_BYTES,
                },
            }));
            return StepResult::Done;
        }

        // Find earliest of the three possible terminators.
        let call_pos = body.find(TOK_CALL);
        let return_pos = body.find(TOK_RETURN);
        let end_pos = body.find(TOK_END);
        let earliest = [
            call_pos.map(|p| (p, BodyTerm::Call)),
            return_pos.map(|p| (p, BodyTerm::Return)),
            end_pos.map(|p| (p, BodyTerm::End)),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|(p, _)| *p);

        match earliest {
            Some((pos, term)) => {
                let (body_text, after_offset) = match term {
                    BodyTerm::Call => (&body[..pos], pos + TOK_CALL.len()),
                    BodyTerm::Return => (&body[..pos], pos + TOK_RETURN.len()),
                    BodyTerm::End => (&body[..pos], pos + TOK_END.len()),
                };
                let after = body[after_offset..].to_string();
                self.commit_body(channel, recipient, body_text.to_string(), term, events);
                if matches!(self.state, State::Terminated) {
                    return StepResult::Done;
                }
                *remaining = after;
                StepResult::Continue
            }
            None => {
                // No terminator yet. For final-channel text, drain the
                // safe-to-emit prefix as TextDelta. For commentary /
                // analysis bodies, just keep buffering.
                if matches!(channel, Channel::Final) {
                    let safe_end = safe_emit_boundary_for_three(
                        &body, TOK_CALL, TOK_RETURN, TOK_END,
                    );
                    if safe_end > 0 {
                        let safe = body.drain(..safe_end).collect::<String>();
                        if !safe.is_empty() {
                            events.push(DecodeEvent::TextDelta(safe));
                        }
                    }
                }
                self.state = State::InBody {
                    channel,
                    recipient,
                    body,
                };
                StepResult::Done
            }
        }
    }

    /// Commit a fully-collected message body. Decides the routing:
    /// emit final-channel tail as TextDelta, parse commentary tool
    /// calls, drop commentary preambles and analysis content. After a
    /// successful commit we transition back to `Outside`.
    fn commit_body(
        &mut self,
        channel: Channel,
        recipient: Option<String>,
        body: String,
        _term: BodyTerm,
        events: &mut Vec<DecodeEvent>,
    ) {
        match channel {
            Channel::Final => {
                if !body.is_empty() {
                    events.push(DecodeEvent::TextDelta(body));
                }
                self.state = State::Outside {
                    outside_buf: String::new(),
                };
            }
            Channel::Commentary => {
                if let Some(name) = recipient_to_function_name(recipient.as_deref()) {
                    let index = self.next_index;
                    match parse_tool_call(&body, &name, index, &self.directory) {
                        ToolOutcome::Call(call_events) => {
                            self.next_index += 1;
                            events.extend(call_events);
                            self.state = State::Outside {
                                outside_buf: String::new(),
                            };
                        }
                        ToolOutcome::Fatal(error_event) => {
                            events.extend(self.fatal(error_event));
                        }
                    }
                } else {
                    // Commentary preamble or addressed to a non-function
                    // recipient. Drop silently — see module doc.
                    self.state = State::Outside {
                        outside_buf: String::new(),
                    };
                }
            }
            Channel::Analysis => {
                // Reasoning content — drop. See module doc.
                self.state = State::Outside {
                    outside_buf: String::new(),
                };
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BodyTerm {
    Call,
    Return,
    End,
}

enum ToolOutcome {
    Call(Vec<DecodeEvent>),
    Fatal(DecodeEvent),
}

/// Parse a channel header like `commentary to=functions.foo <|constrain|>json`
/// into `(channel, recipient)`. Tokens are space-separated; we accept
/// them in any order after the channel name.
fn parse_channel_header(header: &str) -> Result<(Channel, Option<String>), ParserError> {
    let mut tokens = header.split_ascii_whitespace();
    let channel_str = tokens.next().ok_or_else(|| {
        ParserError::Malformed("empty harmony channel header".to_string())
    })?;
    // The channel name itself may have a `<` boundary (per harmony's
    // own parser: read up to whitespace OR `<`). Handle the case where
    // it's followed immediately by `<|constrain|>...` without a space.
    let (channel_name, leftover_after_channel) = match channel_str.find('<') {
        Some(i) => (&channel_str[..i], &channel_str[i..]),
        None => (channel_str, ""),
    };
    let channel = match channel_name {
        "final" => Channel::Final,
        "commentary" => Channel::Commentary,
        "analysis" => Channel::Analysis,
        other if other.is_empty() => {
            return Err(ParserError::Malformed(
                "harmony channel header has empty channel name".to_string(),
            ));
        }
        other => {
            return Err(ParserError::Malformed(format!(
                "unknown harmony channel `{other}`"
            )));
        }
    };
    let mut recipient = None;
    // Process leftover after the channel name (may be a `<|constrain|>...`
    // glued on with no space) plus subsequent space-separated tokens.
    let mut all_tokens: Vec<&str> = Vec::new();
    if !leftover_after_channel.is_empty() {
        all_tokens.push(leftover_after_channel);
    }
    all_tokens.extend(tokens);
    for tok in all_tokens {
        if let Some(rest) = tok.strip_prefix("to=") {
            recipient = Some(rest.to_string());
        }
        // Other tokens (e.g. `<|constrain|>json`) are ignored — we
        // don't enforce content-type at this layer.
    }
    Ok((channel, recipient))
}

/// Strip the `functions.` namespace and return the bare tool name, if
/// the recipient is a function call. `to=functions.add` → `Some("add")`;
/// `to=browser.search` or `None` → `None`.
fn recipient_to_function_name(recipient: Option<&str>) -> Option<String> {
    let r = recipient?;
    let name = r.strip_prefix("functions.")?;
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

fn parse_tool_call(
    body: &str,
    name: &str,
    index: usize,
    directory: &ToolDirectory,
) -> ToolOutcome {
    let trimmed = body.trim();
    let value: JsonValue = if trimmed.is_empty() {
        JsonValue::Object(JsonMap::new())
    } else {
        match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(err) => {
                return ToolOutcome::Fatal(DecodeEvent::ParseError {
                    sentinel: ERR_SENTINEL,
                    source: ParserError::from(err),
                });
            }
        }
    };
    if !matches!(value, JsonValue::Object(_)) {
        return ToolOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: ERR_SENTINEL,
            source: ParserError::Malformed(
                "harmony tool-call body must be a JSON object".to_string(),
            ),
        });
    }
    if directory.lookup(name).is_none() {
        return ToolOutcome::Fatal(DecodeEvent::UnknownTool {
            name: name.to_string(),
            raw_args: value,
        });
    }
    let errors = directory.validate_args(name, &value);
    if !errors.is_empty() {
        return ToolOutcome::Fatal(DecodeEvent::InvalidArgs {
            name: name.to_string(),
            args: value,
            errors,
        });
    }
    let args_text = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
    ToolOutcome::Call(vec![
        DecodeEvent::ToolCallStart {
            index,
            name: name.to_string(),
        },
        DecodeEvent::ToolCallArgsDelta {
            index,
            delta: args_text,
        },
        DecodeEvent::ToolCallEnd {
            index,
            args: value,
        },
    ])
}

/// Largest `i ≤ buf.len()` such that `buf[..i]` cannot start either
/// of the two sentinels and lands on a UTF-8 boundary. Used by the
/// outside-state to decide how much plain text is safe to flush.
fn safe_emit_boundary_for_two(buf: &str, sa: &str, sb: &str) -> usize {
    let limit_a = boundary_for_one(buf, sa);
    let limit_b = boundary_for_one(buf, sb);
    limit_a.min(limit_b)
}

fn safe_emit_boundary_for_three(buf: &str, sa: &str, sb: &str, sc: &str) -> usize {
    let la = boundary_for_one(buf, sa);
    let lb = boundary_for_one(buf, sb);
    let lc = boundary_for_one(buf, sc);
    la.min(lb).min(lc)
}

fn boundary_for_one(buf: &str, sentinel: &str) -> usize {
    if buf.is_empty() {
        return 0;
    }
    let max_take = buf.len().min(sentinel.len().saturating_sub(1));
    for take in (1..=max_take).rev() {
        let cut = buf.len() - take;
        if !buf.is_char_boundary(cut) {
            continue;
        }
        if sentinel.starts_with(&buf[cut..]) {
            return cut;
        }
    }
    buf.len()
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

    fn collect_text(events: &[DecodeEvent]) -> String {
        let mut s = String::new();
        for ev in events {
            if let DecodeEvent::TextDelta(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    #[test]
    fn plain_text_outside_channel_passes_through() {
        // Defensive: gpt-oss should always wrap output in channels,
        // but if a buggy detokenizer emits raw text it must surface
        // gracefully as a TextDelta rather than crash.
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(&mut p, &["just some bare text"]);
        assert_eq!(collect_text(&events), "just some bare text");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn final_channel_emits_text_delta() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|channel|>final<|message|>The answer is 42<|return|>"],
        );
        assert_eq!(collect_text(&events), "The answer is 42");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn final_channel_with_start_prefix_emits_text_delta() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|start|>assistant<|channel|>final<|message|>hello<|return|>"],
        );
        assert_eq!(collect_text(&events), "hello");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn commentary_channel_with_function_tool_emits_tool_call() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.add <|constrain|>json<|message|>{"a":1,"b":2}<|call|>"#,
            ],
        );
        // Start, ArgsDelta, End, Stop.
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { .. }));
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!()
        };
        assert_eq!(args, &json!({"a": 1, "b": 2}));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn commentary_channel_arguments_parsed_as_json() {
        // Distinct test: verify nested JSON body parses correctly.
        let nested_tool = ToolSpec::new(
            "search",
            None,
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "year": { "type": "number" },
                            "tags": { "type": "array", "items": { "type": "string" } }
                        }
                    }
                },
                "required": ["query"]
            }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![nested_tool]).unwrap());
        let mut p = GptOssParser::new(dir);
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.search<|message|>{"query":"x","filters":{"year":2024,"tags":["a","b"]}}<|call|>"#,
            ],
        );
        let DecodeEvent::ToolCallEnd { args, .. } = &events[2] else {
            panic!("got: {events:?}")
        };
        assert_eq!(
            args,
            &json!({"query":"x","filters":{"year":2024,"tags":["a","b"]}})
        );
    }

    #[test]
    fn multiple_tool_calls_in_sequence_via_separate_channels() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.add<|message|>{"a":1,"b":2}<|call|>"#,
                r#"<|start|>assistant<|channel|>commentary to=functions.add<|message|>{"a":3,"b":4}<|call|>"#,
            ],
        );
        // 2 triples + Stop = 7
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
    fn final_then_tool_call_then_final_in_one_stream() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|channel|>final<|message|>let me check<|end|>",
                r#"<|start|>assistant<|channel|>commentary to=functions.add<|message|>{"a":1,"b":2}<|call|>"#,
                "<|start|>assistant<|channel|>final<|message|>done<|return|>",
            ],
        );
        // Order: TextDelta("let me check"), ToolCallStart/Args/End,
        // TextDelta("done"), Stop.
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "let me check"
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == "done"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn analysis_channel_is_dropped() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|channel|>analysis<|message|>thinking hard<|end|>",
                "<|start|>assistant<|channel|>final<|message|>answer<|return|>",
            ],
        );
        // analysis content must not appear; only final's "answer" is text.
        assert_eq!(collect_text(&events), "answer");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn commentary_without_function_recipient_is_dropped() {
        // Commentary preamble (no `to=functions.*`) — neither tool call
        // nor user-visible text. Drop silently.
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|channel|>commentary<|message|>I'll call a tool now<|end|>",
                "<|start|>assistant<|channel|>final<|message|>ok<|return|>",
            ],
        );
        assert_eq!(collect_text(&events), "ok");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unknown_tool_in_commentary_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.delete_db<|message|>{}<|call|>"#,
            ],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn invalid_args_in_commentary_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.add<|message|>{"a":"x"}<|call|>"#,
            ],
        );
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}")
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_in_commentary_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                "<|channel|>commentary to=functions.add<|message|>not json at all<|call|>",
            ],
        );
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unknown_channel_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|channel|>weather<|message|>sunny<|end|>"],
        );
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError")
        };
        assert!(matches!(source, ParserError::Malformed(_)));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn partial_channel_sentinel_split_across_feeds() {
        let mut p = GptOssParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("<|chan"));
        events.extend(p.feed("nel|>final<|message|>hi<|return|>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert_eq!(collect_text(&events), "hi");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_message_sentinel_split_across_feeds() {
        let mut p = GptOssParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed("<|channel|>final<|mess"));
        events.extend(p.feed("age|>hello<|return|>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert_eq!(collect_text(&events), "hello");
    }

    #[test]
    fn partial_call_sentinel_split_across_feeds() {
        let mut p = GptOssParser::new(directory_with_add());
        let mut events = Vec::new();
        events.extend(p.feed(
            r#"<|channel|>commentary to=functions.add<|message|>{"a":1,"b":2}<|ca"#,
        ));
        events.extend(p.feed("ll|>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    #[test]
    fn unterminated_channel_at_eos_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(&mut p, &["<|channel|>final<|message|>incomplete..."]);
        // Final-channel text streams incrementally, so the prefix
        // "incomplete..." is emitted as TextDelta(s) before the
        // unterminated ParseError fires from finish(). The contract is:
        // a ParseError + Stop{ProtocolError} pair must appear at the
        // tail; whether earlier TextDelta events were emitted is
        // implementation-dependent (see `final_channel_streams_text_incrementally`).
        let parse_err = events
            .iter()
            .find_map(|e| match e {
                DecodeEvent::ParseError { source, .. } => Some(source),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected ParseError, got {events:?}"));
        assert!(matches!(parse_err, ParserError::Unterminated));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn unterminated_header_at_eos_is_terminal() {
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(&mut p, &["<|channel|>final<|mess"]);
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::Unterminated,
                ..
            }
        ));
    }

    #[test]
    fn payload_over_limit_is_fatal() {
        let mut p = GptOssParser::new(directory_with_add());
        let oversize = "x".repeat(MAX_BODY_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<|channel|>commentary to=functions.add<|message|>"));
        events.extend(p.feed(&oversize));
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(
            source,
            ParserError::PayloadTooLarge { limit_bytes }
                if *limit_bytes == MAX_BODY_BYTES
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn after_fatal_subsequent_feed_returns_empty() {
        let mut p = GptOssParser::new(directory_with_add());
        let first = p.feed(
            r#"<|channel|>commentary to=functions.delete_db<|message|>{}<|call|>"#,
        );
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(p.feed("more text").is_empty());
        assert!(
            p.feed("<|channel|>final<|message|>x<|return|>")
                .is_empty()
        );
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn final_channel_streams_text_incrementally() {
        // Streaming-claim: a final-channel message's text must drain
        // to TextDelta events as it arrives, not buffer until <|return|>.
        let mut p = GptOssParser::new(directory_with_add());
        let first = p.feed("<|channel|>final<|message|>hello there");
        assert_eq!(collect_text(&first), "hello there");
        let second = p.feed(" friend<|return|>");
        assert!(collect_text(&second).contains("friend"));
    }

    #[test]
    fn final_then_end_terminator_works() {
        // <|end|> is also a valid terminator for final-channel messages
        // (used when persisted to history; some streams may emit it).
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &["<|channel|>final<|message|>persisted<|end|>"],
        );
        assert_eq!(collect_text(&events), "persisted");
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn commentary_with_constrain_token_in_header() {
        // The harmony header may glue <|constrain|>json onto the channel
        // line. We accept and ignore it.
        let mut p = GptOssParser::new(directory_with_add());
        let events = run(
            &mut p,
            &[
                r#"<|channel|>commentary to=functions.add <|constrain|>json<|message|>{"a":1,"b":2}<|call|>"#,
            ],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }


    #[test]
    fn parse_channel_header_extracts_channel_and_recipient() {
        let (ch, recipient) =
            parse_channel_header("commentary to=functions.add <|constrain|>json").unwrap();
        assert_eq!(ch, Channel::Commentary);
        assert_eq!(recipient.as_deref(), Some("functions.add"));
    }

    #[test]
    fn parse_channel_header_final_no_recipient() {
        let (ch, recipient) = parse_channel_header("final").unwrap();
        assert_eq!(ch, Channel::Final);
        assert_eq!(recipient, None);
    }

    #[test]
    fn recipient_to_function_name_strips_namespace() {
        assert_eq!(
            recipient_to_function_name(Some("functions.add")).as_deref(),
            Some("add")
        );
        assert_eq!(recipient_to_function_name(Some("browser.search")), None);
        assert_eq!(recipient_to_function_name(None), None);
        assert_eq!(recipient_to_function_name(Some("functions.")), None);
    }

    #[test]
    fn empty_args_object_in_commentary_is_accepted() {
        let no_args_tool = ToolSpec::new(
            "ping",
            None,
            json!({ "type": "object", "properties": {} }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![no_args_tool]).unwrap());
        let mut p = GptOssParser::new(dir);
        let events = run(
            &mut p,
            &["<|channel|>commentary to=functions.ping<|message|>{}<|call|>"],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
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
            // plain bare text (defensive)
            "hello world",
            // final channel only
            "<|channel|>final<|message|>The answer is 42<|return|>",
            // final with start prefix
            "<|start|>assistant<|channel|>final<|message|>hello<|return|>",
            // single tool call
            r#"<|channel|>commentary to=functions.add<|message|>{"a":1,"b":2}<|call|>"#,
            // tool call with constrain
            r#"<|channel|>commentary to=functions.add <|constrain|>json<|message|>{"a":1,"b":2}<|call|>"#,
            // analysis then final
            "<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>",
            // tool call then final
            r#"<|channel|>commentary to=functions.add<|message|>{"a":1,"b":2}<|call|><|start|>assistant<|channel|>final<|message|>done<|return|>"#,
            // unknown tool (fatal — chunk-invariance still holds)
            r#"<|channel|>commentary to=functions.missing<|message|>{}<|call|>"#,
            // commentary preamble (dropped)
            "<|channel|>commentary<|message|>about to call<|end|><|start|>assistant<|channel|>final<|message|>x<|return|>",
            // sentinel-shaped text outside (defensive)
            "the docs say <|channel|> but it's just text",
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..10,
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
            input_idx in 0_usize..10,
            mut splits in prop::collection::vec(0_usize..400, 1..6),
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
