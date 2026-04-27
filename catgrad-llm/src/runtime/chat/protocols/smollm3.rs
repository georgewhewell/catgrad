//! SmolLM3-Instruct tool-call protocol.
//!
//! # Wire format
//!
//! Hermes-style sentinel-wrapped JSON, identical to Qwen3 / SmolLM2:
//!
//! ```text
//! preamble text<tool_call>{"name": "x", "arguments": {...}}</tool_call>
//! more text<tool_call>{"name": "y", "arguments": {...}}</tool_call>
//! ```
//!
//! SmolLM3 reserves dedicated `<tool_call>` / `</tool_call>` tokens
//! (vocabulary IDs 128015 / 128016) so the model emits them as single
//! tokens rather than multi-token sequences — but the parser can still
//! match them as plain text, so we delegate to the shared
//! [`super::json_sentinel`] state machine.
//!
//! # Why a separate protocol from Qwen3
//!
//! Functionally the wire-level parser is identical; the differences
//! that matter for production are:
//!
//! 1. **Architecture string.** SmolLM3 reports `SmolLM3ForCausalLM` in
//!    `config.json`, distinct from Qwen3 and from the
//!    `LlamaForCausalLM`-shaped models — so the lookup is
//!    unambiguous and doesn't need a tokenizer fingerprint.
//!
//! 2. **Parallel calls.** SmolLM3's chat template iterates a list of
//!    tool calls per assistant turn, so parallel tool calling is
//!    supported — exposed via `supports_parallel_calls = true`.
//!
//! 3. **Chat template behaviour.** SmolLM3's template natively
//!    iterates `tools` (xml_tools / python_tools) and prepends the
//!    full tool-call format instructions to the system message. No
//!    [`prepare_messages`] injection is needed — identity over the
//!    message list.

use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::runtime::chat::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use crate::types;

use super::json_sentinel;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Construct a SmolLM3 parser bound to the given tool directory.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    json_sentinel::make_parser(directory, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
}

/// Render the bound tool list into the OpenAI-style envelope SmolLM3's
/// chat template expects — same shape as Qwen3 / SmolLM2.
pub fn render_tools(specs: &[ToolSpec]) -> JsonValue {
    json_sentinel::render_openai_tool_envelope(specs)
}

/// SmolLM3's chat template renders the tool spec into the system
/// preamble itself, so the protocol does not inject anything. Identity
/// over the message list.
pub fn prepare_messages(_specs: &[ToolSpec], messages: Vec<types::Message>) -> Vec<types::Message> {
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{
        DecodeEvent, IncrementalToolCallParser, StopReason, ToolDirectory, ToolSpec,
    };
    use serde_json::json;

    fn add_tool() -> ToolSpec {
        ToolSpec::new(
            "add",
            None,
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
            }),
        )
    }

    #[test]
    fn parser_uses_same_wire_format_as_qwen3() {
        let dir = Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap());
        let mut p = make_parser(dir);
        let mut events = p.feed(
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
        );
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    #[test]
    fn prepare_messages_is_identity() {
        let user = types::Message::OpenAI(Box::new(types::openai::ChatMessage::user("hi")));
        let messages = prepare_messages(&[add_tool()], vec![user.clone()]);
        assert_eq!(messages.len(), 1);
        assert_eq!(&messages[0], &user);
    }
}
