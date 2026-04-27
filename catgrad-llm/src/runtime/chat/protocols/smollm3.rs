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

// All production wiring lives in `protocol.rs`'s `SMOLLM3` registry
// entry. The dialect is `<tool_call>JSON</tool_call>` with the
// Hermes-permissive codec — same as Qwen3.

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use crate::runtime::chat::{IncrementalToolCallParser, ToolDirectory};

#[cfg(test)]
fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    use crate::runtime::chat::tool_protocol_for;
    tool_protocol_for("SmolLM3ForCausalLM", &serde_json::Value::Null)
        .expect("SmolLM3 arch is registered")
        .make_parser(directory)
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

    // The identity-prepare_messages contract is covered by the
    // shared `render::identity_prepare_messages` test — no need to
    // duplicate per-protocol.
}
