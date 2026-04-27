//! Shared `render_tools` / `prepare_messages` helpers used by most
//! protocols. The bulk of in-tree dialects (qwen3, smollm2/3, granite,
//! mistral3, phi4, nemotron, olmo3, gemma4, qwen3_5) all render the
//! same OpenAI-style tool envelope and have an identity
//! `prepare_messages` — the registry points each at the relevant
//! function from this module rather than each protocol redefining it.
//!
//! Only protocols whose chat templates demand a different rendering
//! (e.g. lfm2's flat list, smollm2's system-prompt injection) provide
//! their own functions.

use serde_json::{Map as JsonMap, Value as JsonValue};

use super::tool_spec::ToolSpec;
use crate::types;

/// OpenAI-style tool list:
/// `[{"type":"function","function":{"name":..,"description":..,"parameters":..}}]`.
///
/// This is the canonical shape that
/// `transformers.apply_chat_template` produces from a Python tool
/// callable, so emitting it here keeps Rust-rendered prompts
/// byte-identical to what the model was fine-tuned against (for the
/// dialects that pass `tools` straight through `tojson`).
pub fn openai_tool_envelope(specs: &[ToolSpec]) -> JsonValue {
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

/// `prepare_messages` for protocols whose chat template handles
/// `tools` natively — no system-prompt injection needed. Identity.
pub fn identity_prepare_messages(
    _specs: &[ToolSpec],
    messages: Vec<types::Message>,
) -> Vec<types::Message> {
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_tool_envelope_has_canonical_shape() {
        let spec = ToolSpec::new(
            "add",
            Some("add two numbers".into()),
            json!({
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            }),
        );
        let rendered = openai_tool_envelope(&[spec]);
        let arr = rendered.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], json!("function"));
        assert_eq!(arr[0]["function"]["name"], json!("add"));
        assert_eq!(arr[0]["function"]["description"], json!("add two numbers"));
        assert!(arr[0]["function"]["parameters"].is_object());
    }

    #[test]
    fn openai_tool_envelope_omits_missing_description() {
        let spec = ToolSpec::new(
            "x",
            None,
            json!({"type": "object", "properties": {}}),
        );
        let rendered = openai_tool_envelope(&[spec]);
        assert!(rendered.as_array().unwrap()[0]["function"]
            .get("description")
            .is_none());
    }

    #[test]
    fn identity_prepare_messages_returns_input_unchanged() {
        use crate::types;
        let user = types::Message::OpenAI(Box::new(types::openai::ChatMessage::user("hi")));
        let out = identity_prepare_messages(&[], vec![user.clone()]);
        assert_eq!(out, vec![user]);
    }
}
