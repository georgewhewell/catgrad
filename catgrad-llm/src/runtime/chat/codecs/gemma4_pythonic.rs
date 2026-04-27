//! Gemma 4 tool-call codec.
//!
//! Wire shape (between sentinels): `call:NAME{key:value, key2:value2}`.
//! Strings are framed with paired `<|"|>` sentinels (rather than ASCII
//! quotes); this lets the model distinguish argument-string content
//! from JSON-shaped content unambiguously.
//!
//! Detects the outdated chat-template revision that emits
//! `{{...JSON...}}` (HF discussions #20/#55) and surfaces a specific
//! error so operators can pin the offending template.

use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

use super::super::event::ParserError;
use super::super::sentinel_engine::{CodecOutcome, DecodedCall, PayloadCodec};

const STRING_QUOTE: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

pub struct Gemma4Codec;

impl PayloadCodec for Gemma4Codec {
    fn parse(&self, payload: &str) -> CodecOutcome {
        let trimmed = payload.trim();
        if trimmed.is_empty() {
            return CodecOutcome::Error(ParserError::Malformed(
                "empty tool-call payload".into(),
            ));
        }
        // HF discussions #20/#55 on google/gemma-4-*-it: an outdated
        // chat-template revision emits `{{...JSON...}}` instead of the
        // bare-key form. Detect it before the `call:` strip so the
        // operator gets a useful hint pointing at the upstream issue.
        if trimmed.starts_with("{{") {
            return CodecOutcome::Error(ParserError::Malformed(
                "tool-call payload has double-braced JSON body — \
                 the chat template is an outdated revision (see \
                 huggingface.co/google/gemma-4-*-it discussions #20/#55)"
                    .into(),
            ));
        }
        let Some(rest) = trimmed.strip_prefix(CALL_PREFIX) else {
            return CodecOutcome::Error(ParserError::Malformed(
                "missing `call:` prefix in tool-call payload".into(),
            ));
        };
        let Some(open_brace) = rest.find('{') else {
            return CodecOutcome::Error(ParserError::Malformed(
                "tool-call payload missing `{` after function name".into(),
            ));
        };
        let name = rest[..open_brace].trim();
        if name.is_empty() {
            return CodecOutcome::Error(ParserError::MissingField("name"));
        }
        if !is_valid_function_name(name) {
            return CodecOutcome::Error(ParserError::Malformed(format!(
                "invalid function name `{name}` — must match [A-Za-z_][A-Za-z0-9_\\-\\.]*"
            )));
        }
        let body = rest[open_brace..].trim_end();
        if !body.ends_with('}') {
            return CodecOutcome::Error(ParserError::Malformed(
                "tool-call payload missing closing `}`".into(),
            ));
        }
        let inner = &body[1..body.len() - 1];
        let args = match parse_object_body(inner) {
            Ok(value) => value,
            Err(message) => return CodecOutcome::Error(ParserError::Malformed(message)),
        };
        // Defensive: a literal `<|"|>` surviving means upstream
        // detokenization stripped one half of a pair. Surface as a
        // hard error rather than ship a bogus call.
        if args_contain_literal_quote_sentinel(&args) {
            return CodecOutcome::Error(ParserError::Malformed(
                "parsed args still contain a literal `<|\"|>` sentinel".into(),
            ));
        }
        CodecOutcome::Calls(vec![DecodedCall {
            name: name.to_string(),
            args,
        }])
    }
}

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

fn args_contain_literal_quote_sentinel(value: &JsonValue) -> bool {
    match value {
        JsonValue::String(s) => s.contains(STRING_QUOTE),
        JsonValue::Array(items) => items.iter().any(args_contain_literal_quote_sentinel),
        JsonValue::Object(map) => map.values().any(args_contain_literal_quote_sentinel),
        _ => false,
    }
}

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
            return Err(format!("trailing characters after string literal: `{after}`"));
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

// Bracket / quote / separator walking is delegated to the shared
// [`super::balanced_lexer`] with the Gemma 4 config (paired
// `<|"|>` strings, no parens). Per-protocol code shrinks to the
// narrow Gemma-specific bits (the `call:` prefix and `<|"|>`-quoted
// string handling done at value level).
fn split_top_level(text: &str, separator: char) -> Result<Vec<&str>, String> {
    let mut cfg = super::balanced_lexer::BalancedConfig::GEMMA4;
    cfg.separator = separator;
    super::balanced_lexer::split_top_level(text, &cfg)
}

fn find_top_level_colon(text: &str) -> Result<Option<usize>, String> {
    super::balanced_lexer::find_top_level(
        text,
        ':',
        &super::balanced_lexer::BalancedConfig::GEMMA4,
    )
}
