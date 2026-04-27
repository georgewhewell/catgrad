//! Per-architecture tool-call capability registry.
//!
//! Each entry describes how to render the bound tool list for the chat
//! template, how to massage the message list before rendering, and how
//! to construct an incremental parser bound to a [`ToolDirectory`].
//!
//! [`tool_protocol_for`] is the single entry point. Architectures
//! missing from the table cannot serve tool-enabled chat requests:
//! [`ChatTurn::new`](super::ChatTurn::new) consults this registry and
//! rejects the turn at construction time when tools are bound but no
//! protocol is registered.
//!
//! Some architectures share an HF `architectures[0]` string but use
//! different tool-call dialects (notably: SmolLM2 reports
//! `LlamaForCausalLM` like vanilla Llama / Mistral, but uses ChatML
//! tokens and the Hermes-style `<tool_call>` format). The dispatch
//! takes the model's `tokenizer_config.json` so it can disambiguate
//! using the tokenizer fingerprint without relying on model-name
//! string matching.
//!
//! Adding support for a new architecture means writing one
//! [`IncrementalToolCallParser`] state machine, a `render_tools` shaping
//! function, an optional `prepare_messages` hook, and one row in
//! [`tool_protocol_for`].

use std::sync::Arc;

use super::protocols;
use super::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use crate::types;
use serde_json::Value as JsonValue;

/// Capability descriptor for one model architecture's tool-calling
/// dialect. Stored as a `&'static` to allow callers to compare protocol
/// identity by pointer when useful.
#[derive(Debug)]
pub struct ToolCallProtocol {
    /// Shape the bound tool list into the JSON value the chat template
    /// expects. Output is what gets bound to the template's `tools`
    /// variable; shape is architecture-specific.
    pub render_tools: fn(&[ToolSpec]) -> JsonValue,

    /// Construct an incremental parser that owns its tool directory.
    /// The returned parser is `'static`, so a caller can hold both the
    /// `ChatTurn` and the parser together (e.g. on a per-request struct
    /// in a gateway) without a self-referential borrow.
    pub make_parser: fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,

    /// Whether the model can emit multiple tool calls in a single
    /// generation. Surfaces to clients via the gateway as the
    /// `parallel_tool_calls` capability.
    pub supports_parallel_calls: bool,

    /// Transform the message list before it is rendered through the
    /// chat template. Used by protocols whose chat template does not
    /// natively iterate `tools` (e.g. SmolLM2's plain ChatML loop): the
    /// protocol injects a system message describing the tools and the
    /// expected wire format. Default for tool-aware templates (Qwen3):
    /// identity.
    pub prepare_messages: fn(&[ToolSpec], Vec<types::Message>) -> Vec<types::Message>,
}

const QWEN3: ToolCallProtocol = ToolCallProtocol {
    render_tools: protocols::qwen3::render_tools,
    make_parser: protocols::qwen3::make_parser,
    supports_parallel_calls: true,
    prepare_messages: protocols::qwen3::prepare_messages,
};

const SMOLLM2: ToolCallProtocol = ToolCallProtocol {
    render_tools: protocols::smollm2::render_tools,
    make_parser: protocols::smollm2::make_parser,
    // Small SmolLM2 instruct fine-tunes have not been validated as
    // reliable parallel-call emitters; expose them as serial only.
    supports_parallel_calls: false,
    prepare_messages: protocols::smollm2::prepare_messages,
};

const SMOLLM3: ToolCallProtocol = ToolCallProtocol {
    render_tools: protocols::smollm3::render_tools,
    make_parser: protocols::smollm3::make_parser,
    supports_parallel_calls: true,
    prepare_messages: protocols::smollm3::prepare_messages,
};

const LLAMA3: ToolCallProtocol = ToolCallProtocol {
    render_tools: protocols::llama3::render_tools,
    make_parser: protocols::llama3::make_parser,
    // Llama 3's chat template raises if `tool_calls | length != 1`,
    // so the dialect is structurally single-call per turn.
    supports_parallel_calls: false,
    prepare_messages: protocols::llama3::prepare_messages,
};

/// Lookup table from `(arch, tokenizer_config)` to the architecture's
/// tool-call protocol. Returns `None` for architectures that do not
/// support tool calling (or that have not yet been ported to the
/// incremental parser model).
///
/// `tokenizer_config` is the parsed `tokenizer_config.json` for the
/// model. Most arches dispatch on `arch` alone; some share an `arch`
/// string and need a tokenizer fingerprint to disambiguate. The
/// caller should pass the same value it will hand to the chat
/// template — pass `&JsonValue::Null` if no tokenizer config is
/// available (the dispatch then falls back to arch-only matching).
pub fn tool_protocol_for(
    arch: &str,
    tokenizer_config: &JsonValue,
) -> Option<&'static ToolCallProtocol> {
    match arch {
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Some(&QWEN3),
        "SmolLM3ForCausalLM" => Some(&SMOLLM3),
        "LlamaForCausalLM" => match extract_bos_token(tokenizer_config) {
            // SmolLM2-Instruct: ChatML tokens, no native tools in
            // template — protocol injects a system prompt.
            Some("<|im_start|>") => Some(&SMOLLM2),
            // Llama 3.x / 4.x Instruct: `<|begin_of_text|>` is the
            // tokenizer fingerprint shared across the meta-llama
            // family that supports the bare-JSON tool dialect.
            Some("<|begin_of_text|>") => Some(&LLAMA3),
            // Vanilla Llama 1/2 / Mistral: `<s>` BOS, no native tool
            // dialect. Fall through to the unsupported path.
            _ => None,
        },
        _ => None,
    }
}

fn extract_bos_token(tokenizer_config: &JsonValue) -> Option<&str> {
    let bos = tokenizer_config.get("bos_token")?;
    bos.as_str()
        .or_else(|| bos.get("content").and_then(JsonValue::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn qwen3_architectures_resolve_to_protocol() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("Qwen3ForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("Qwen3MoeForCausalLM", &cfg).is_some());
    }

    #[test]
    fn unknown_architecture_returns_none() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("Qwen3_5ForConditionalGeneration", &cfg).is_none());
        assert!(tool_protocol_for("Lfm2ForCausalLM", &cfg).is_none());
        assert!(tool_protocol_for("Olmo3ForCausalLM", &cfg).is_none());
        assert!(tool_protocol_for("", &cfg).is_none());
    }

    #[test]
    fn vanilla_llama_with_default_bos_returns_none() {
        // <s> is the standard Llama 1 / 2 / Mistral BOS; no native
        // tool dialect.
        let cfg = json!({ "bos_token": "<s>" });
        assert!(tool_protocol_for("LlamaForCausalLM", &cfg).is_none());
    }

    #[test]
    fn llama_with_no_tokenizer_config_returns_none() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("LlamaForCausalLM", &cfg).is_none());
    }

    #[test]
    fn smollm2_chatml_llama_resolves_to_smollm2_protocol() {
        // SmolLM2-Instruct's tokenizer_config.json has
        // bos_token = `<|im_start|>`.
        let cfg = json!({ "bos_token": "<|im_start|>" });
        let proto = tool_protocol_for("LlamaForCausalLM", &cfg)
            .expect("SmolLM2-style ChatML llama must resolve to SmolLM2 protocol");
        assert_smollm2_protocol(proto);
    }

    #[test]
    fn smollm2_object_bos_token_resolves() {
        // Some tokenizer_configs encode bos_token as an object
        // `{ "content": "<|im_start|>", ... }`.
        let cfg = json!({
            "bos_token": {
                "content": "<|im_start|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false
            }
        });
        let proto = tool_protocol_for("LlamaForCausalLM", &cfg)
            .expect("SmolLM2-style object bos_token must still resolve");
        assert_smollm2_protocol(proto);
    }

    #[test]
    fn smollm3_arch_resolves_to_smollm3_protocol() {
        // SmolLM3 has its own arch string, dispatch is unambiguous.
        let cfg = json!({ "bos_token": "<|begin_of_text|>" });
        let proto = tool_protocol_for("SmolLM3ForCausalLM", &cfg)
            .expect("SmolLM3 must resolve to a tool protocol");
        // Identify by behavior: identity prepare_messages + Hermes
        // sentinel parser (which validates payload as JSON).
        assert_identity_prepare_messages(proto);
        assert!(proto.supports_parallel_calls);
        assert_hermes_style_parser(proto);
    }

    #[test]
    fn llama3_bos_token_resolves_to_llama3_protocol() {
        // Llama 3.x / 4.x Instruct fingerprint: bos = `<|begin_of_text|>`
        // on `LlamaForCausalLM`.
        let cfg = json!({ "bos_token": "<|begin_of_text|>" });
        let proto = tool_protocol_for("LlamaForCausalLM", &cfg)
            .expect("Llama 3 Instruct must resolve to Llama 3 protocol");
        assert_identity_prepare_messages(proto);
        assert!(!proto.supports_parallel_calls);
        // Llama 3 dialect: bare JSON, no `<tool_call>` sentinel.
        assert_bare_json_parser(proto);
    }

    fn assert_smollm2_protocol(proto: &'static ToolCallProtocol) {
        let spec = ToolSpec::new(
            "ping",
            None,
            json!({ "type": "object", "properties": {} }),
        );
        let messages = (proto.prepare_messages)(&[spec], Vec::new());
        assert_eq!(
            messages.len(),
            1,
            "SmolLM2 protocol must inject a system message; got {messages:?}"
        );
        let types::Message::OpenAI(first) = &messages[0] else {
            panic!("expected OpenAI message");
        };
        assert_eq!(first.role, "system");
        assert!(!proto.supports_parallel_calls);
    }

    fn assert_identity_prepare_messages(proto: &'static ToolCallProtocol) {
        let spec = ToolSpec::new(
            "ping",
            None,
            json!({ "type": "object", "properties": {} }),
        );
        let user = crate::types::Message::OpenAI(Box::new(
            crate::types::openai::ChatMessage::user("hi"),
        ));
        let out = (proto.prepare_messages)(&[spec], vec![user.clone()]);
        assert_eq!(out, vec![user]);
    }

    /// A Hermes-style protocol parser must reject bare JSON without a
    /// `<tool_call>` sentinel — they pass through as plain text.
    fn assert_hermes_style_parser(proto: &'static ToolCallProtocol) {
        let dir = make_test_directory();
        let mut parser = (proto.make_parser)(dir);
        let bare_json = r#"{"name":"ping","arguments":{}}"#;
        let mut events = parser.feed(bare_json);
        events.extend(parser.finish(super::super::StopReason::EndOfText));
        let has_tool_call = events
            .iter()
            .any(|e| matches!(e, super::super::DecodeEvent::ToolCallStart { .. }));
        assert!(
            !has_tool_call,
            "Hermes parsers must not treat bare JSON as a tool call; got {events:?}"
        );
    }

    /// A bare-JSON protocol parser must accept `{...}` (no sentinel)
    /// as a tool call.
    fn assert_bare_json_parser(proto: &'static ToolCallProtocol) {
        let dir = make_test_directory();
        let mut parser = (proto.make_parser)(dir);
        let bare_json = r#"{"name":"ping","parameters":{}}"#;
        let mut events = parser.feed(bare_json);
        events.extend(parser.finish(super::super::StopReason::EndOfText));
        let has_tool_call = events
            .iter()
            .any(|e| matches!(e, super::super::DecodeEvent::ToolCallStart { .. }));
        assert!(
            has_tool_call,
            "bare-JSON parsers must accept sentinel-less JSON as a tool call; got {events:?}"
        );
    }

    fn make_test_directory() -> Arc<ToolDirectory> {
        Arc::new(
            ToolDirectory::new(vec![ToolSpec::new(
                "ping",
                None,
                json!({ "type": "object", "properties": {}, "additionalProperties": true }),
            )])
            .unwrap(),
        )
    }
}
