//! XML-style function-call codec.
//!
//! Wire shape (between sentinels):
//!
//! ```text
//! <function=NAME>
//!   <parameter=KEY1>VALUE1</parameter>
//!   <parameter=KEY2>VALUE2</parameter>
//! </function>
//! ```
//!
//! Used by Qwen3.5 and Nemotron / Nemotron-H — vLLM's
//! `qwen3_xml_tool_parser.py` and `nemotron_h_tool_parser.py` are the
//! reference implementations. Both protocols emit at most one call per
//! sentinel block.
//!
//! Parameter values: try JSON first (numbers, booleans, null, nested
//! arrays/objects); fall back to a JSON string literal of the trimmed
//! text. Mirrors vLLM's `parse_scalar`.

use serde_json::{Map as JsonMap, Value as JsonValue};

use super::super::event::ParserError;
use super::super::sentinel_engine::{CodecOutcome, DecodedCall, PayloadCodec};

const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

pub struct XmlFunctionCodec;

impl PayloadCodec for XmlFunctionCodec {
    fn parse(&self, payload: &str) -> CodecOutcome {
        let trimmed = payload.trim();
        if trimmed.is_empty() {
            return CodecOutcome::Error(ParserError::Malformed(
                "empty tool-call payload".into(),
            ));
        }
        match parse_function_block(trimmed) {
            Ok((name, args)) => CodecOutcome::Calls(vec![DecodedCall { name, args }]),
            Err(err) => CodecOutcome::Error(err),
        }
    }
}

fn parse_function_block(payload: &str) -> Result<(String, JsonValue), ParserError> {
    let function_start = payload.find(FUNCTION_OPEN).ok_or_else(|| {
        ParserError::Malformed("tool-call payload missing <function=...> block".into())
    })?;
    let header = &payload[function_start + FUNCTION_OPEN.len()..];
    let name_end = header
        .find('>')
        .ok_or_else(|| ParserError::Malformed("unterminated <function=...> tag".into()))?;
    let name = header[..name_end].trim().to_string();
    if name.is_empty() {
        return Err(ParserError::MissingField("name"));
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

fn parse_scalar(text: &str) -> JsonValue {
    let trimmed = text.trim();
    serde_json::from_str(trimmed).unwrap_or_else(|_| JsonValue::String(trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        assert_eq!(parse_scalar("not json"), json!("not json"));
    }

    #[test]
    fn parse_function_block_extracts_name_and_args() {
        let (name, args) = parse_function_block(
            "<function=add><parameter=a>1</parameter><parameter=b>2</parameter></function>",
        )
        .unwrap();
        assert_eq!(name, "add");
        assert_eq!(args["a"], json!(1));
        assert_eq!(args["b"], json!(2));
    }

    #[test]
    fn parse_function_block_errors_on_missing_function_open() {
        assert!(parse_function_block("just text").is_err());
    }

    #[test]
    fn parse_function_block_errors_on_unclosed_function() {
        assert!(parse_function_block("<function=x><parameter=k>v</parameter>").is_err());
    }

    #[test]
    fn parse_function_block_errors_on_empty_name() {
        assert!(matches!(
            parse_function_block("<function=></function>"),
            Err(ParserError::MissingField("name"))
        ));
    }
}
