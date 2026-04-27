//! Pythonic call-list payload codec.
//!
//! Wire shape: `name1(k=v, k2=v2), name2(k=...)` — Python function-call
//! literals separated by commas at the top level. The whole payload may
//! optionally be wrapped in `[...]`. Used by olmo3, by lfm2 (which sniffs
//! between this and the JSON codec), and by Gemma 4 (with its own
//! per-string framing — Gemma uses a separate codec because its values
//! aren't standard Python literals).
//!
//! Value lexer concessions, matching the in-tree behavior of the
//! reference vLLM/SGLang Pythonic parsers:
//! - `True` / `False` / `None` → JSON bool / null.
//! - JSON-shaped values (strings, numbers, arrays, objects) parse as
//!   themselves.
//! - Anything else → bare-string fallback (unquoted enum-like values
//!   are common in Pythonic outputs).
//!
//! Empty payload is **fatal** here, mirroring olmo3. If a future caller
//! wants to tolerate `[]` as zero-calls, parameterize the codec.

use serde_json::{Map as JsonMap, Value as JsonValue};

use super::super::event::ParserError;
use super::super::sentinel_engine::{CodecOutcome, DecodedCall, PayloadCodec};

pub struct PythonicCallsCodec;

impl PayloadCodec for PythonicCallsCodec {
    fn parse(&self, payload: &str) -> CodecOutcome {
        let trimmed = payload.trim();
        if trimmed.is_empty() {
            return CodecOutcome::Error(ParserError::Malformed(
                "empty tool-call payload".into(),
            ));
        }
        match parse_python_calls(trimmed) {
            Ok(calls) => {
                if calls.is_empty() {
                    return CodecOutcome::Error(ParserError::Malformed(
                        "tool-call payload contained no calls".into(),
                    ));
                }
                CodecOutcome::Calls(
                    calls
                        .into_iter()
                        .map(|(name, args)| DecodedCall { name, args })
                        .collect(),
                )
            }
            Err(err) => CodecOutcome::Error(err),
        }
    }
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

pub(crate) fn parse_python_calls(text: &str) -> Result<Vec<(String, JsonValue)>, ParserError> {
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
            // Matches the reference Pythonic parsers' permissiveness on
            // unquoted enum-like values.
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
pub(crate) fn split_top_level(text: &str, separator: char) -> Result<Vec<&str>, ParserError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn split_top_level_respects_nested_brackets() {
        let parts = split_top_level("a(1, 2), b([3, 4]), c", ',').unwrap();
        assert_eq!(parts, vec!["a(1, 2)", "b([3, 4])", "c"]);
    }

    #[test]
    fn split_top_level_drops_empty_parts() {
        let parts = split_top_level("a,, b", ',').unwrap();
        assert_eq!(parts, vec!["a", "b"]);
    }

    #[test]
    fn split_top_level_errors_on_unbalanced_brackets() {
        assert!(split_top_level("a(1", ',').is_err());
    }

    #[test]
    fn find_top_level_char_skips_inside_strings() {
        // The `=` inside the quoted string must not be the top-level match;
        // the second `=` (between `b` and `2`) is the one we want.
        assert_eq!(find_top_level_char(r#"a="x=y", b=2"#, '='), Some(1));
    }
}

pub(crate) fn find_top_level_char(text: &str, needle: char) -> Option<usize> {
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
