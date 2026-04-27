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
use super::render;
use super::sentinel_engine::{PayloadCodec, SentinelEngine, SentinelKind};
use super::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use crate::types;
use serde_json::Value as JsonValue;

/// How the protocol's parser is constructed. The vast majority of
/// in-tree dialects fit [`Self::Engine`] — sentinel-bounded payload
/// run through a pluggable [`PayloadCodec`]. The exceptions
/// ([`protocols::gpt_oss`] harmony channels, [`protocols::llama3`]
/// bare-JSON streaming) carry a [`Self::Custom`] constructor so they
/// can implement [`IncrementalToolCallParser`] directly.
#[derive(Clone, Copy)]
pub enum ParserShape {
    /// Sentinel-bounded protocol. The registry stamps out a fresh
    /// [`SentinelEngine`] per request, holding the sentinel
    /// description and a fresh codec instance.
    Engine {
        sentinel: SentinelKind,
        codec_factory: fn() -> Box<dyn PayloadCodec>,
    },
    /// Hand-rolled parser. The function returns a fresh
    /// `Box<dyn IncrementalToolCallParser>` per request.
    Custom(fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>),
}

impl std::fmt::Debug for ParserShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine { sentinel, .. } => f
                .debug_struct("Engine")
                .field("sentinel", sentinel)
                .field("codec_factory", &"<fn>")
                .finish(),
            Self::Custom(_) => f.write_str("Custom(<fn>)"),
        }
    }
}

/// Wire-encoded examples used by the shared protocol-test harness in
/// `protocol_test_kit`. Each `&'static str` is a valid input for the
/// corresponding scenario, encoded in the protocol's wire format. The
/// harness asserts the resulting `DecodeEvent` shape; the protocol
/// just declares its dialect.
///
/// `None` opts the protocol out of the universal harness. Used by
/// the structural outliers (`llama3` bare-JSON streaming and
/// `gpt_oss` harmony channels) — their wire shape doesn't match the
/// "sentinel-bounded valid/unknown/invalid/malformed payload" model
/// the universal scenarios assume.
#[derive(Debug, Clone, Copy)]
pub struct ProtocolExamples {
    /// Wire-encoded call to `add` with `{a: 1, b: 2}`. Universal
    /// directory in the harness binds an `add(a: number, b: number)`
    /// tool, so this should resolve cleanly to a valid triple.
    pub valid_call_add_1_2: &'static str,
    /// Wire-encoded call to a tool name NOT in the harness's
    /// directory. Should fatal-out with `UnknownTool`.
    pub unknown_tool: &'static str,
    /// Wire-encoded call to `add` with args that violate the schema
    /// (e.g. `a` is a string). Should fatal-out with `InvalidArgs`.
    pub invalid_args: &'static str,
    /// Sentinel-opened block whose payload is malformed in the
    /// protocol's dialect. Should fatal-out with `ParseError`.
    pub malformed_payload: &'static str,
    /// Just the open sentinel string — the harness uses this to
    /// start a block then stuff an oversize body, asserting the
    /// payload-too-large error.
    pub open_sentinel_only: &'static str,
}

/// Capability descriptor for one model architecture's tool-calling
/// dialect. Stored as a `&'static` to allow callers to compare protocol
/// identity by pointer when useful.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallProtocol {
    /// Shape the bound tool list into the JSON value the chat template
    /// expects. Output is what gets bound to the template's `tools`
    /// variable; shape is architecture-specific. Most dialects use
    /// [`render::openai_tool_envelope`].
    pub render_tools: fn(&[ToolSpec]) -> JsonValue,

    /// How to construct the protocol's parser. See [`ParserShape`].
    pub parser: ParserShape,

    /// Whether the model can emit multiple tool calls in a single
    /// generation. Surfaces to clients via the gateway as the
    /// `parallel_tool_calls` capability.
    pub supports_parallel_calls: bool,

    /// Transform the message list before it is rendered through the
    /// chat template. Used by protocols whose chat template does not
    /// natively iterate `tools` (e.g. SmolLM2's plain ChatML loop): the
    /// protocol injects a system message describing the tools and the
    /// expected wire format. Most dialects use
    /// [`render::identity_prepare_messages`].
    pub prepare_messages: fn(&[ToolSpec], Vec<types::Message>) -> Vec<types::Message>,

    /// Wire-encoded examples for the universal protocol-test harness.
    /// `None` opts the protocol out (used by the structural outliers
    /// that don't fit the sentinel-bounded model).
    pub examples: Option<ProtocolExamples>,
}

impl ToolCallProtocol {
    /// Construct an incremental parser bound to the given directory.
    /// Single entry point for callers (gateway, examples) — they don't
    /// need to know whether the protocol uses [`SentinelEngine`] or a
    /// custom hand-rolled parser.
    pub fn make_parser(
        &self,
        directory: Arc<ToolDirectory>,
    ) -> Box<dyn IncrementalToolCallParser> {
        match self.parser {
            ParserShape::Engine {
                sentinel,
                codec_factory,
            } => Box::new(SentinelEngine::new(directory, codec_factory(), sentinel)),
            ParserShape::Custom(make) => make(directory),
        }
    }
}

// All 13 in-tree protocols emit the same OpenAI tool-list envelope —
// `render::openai_tool_envelope` is the universal `render_tools` here.
//
// 12 of 13 use identity `prepare_messages`; SmolLM2 is the only
// dialect that injects a system prompt (its chat template ignores
// `tools`).
//
// Sentinel-bounded protocols use `ParserShape::Engine` with the
// appropriate codec; the structurally-different ones (gpt_oss
// harmony channels, llama3 bare-JSON streaming) carry their own
// `ParserShape::Custom` constructor.

const QWEN3: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<tool_call>",
            close: "</tool_call>",
        },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::permissive())
        },
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"##,
        unknown_tool: r##"<tool_call>{"name":"missing","arguments":{}}</tool_call>"##,
        invalid_args: r##"<tool_call>{"name":"add","arguments":{"a":"x","b":2}}</tool_call>"##,
        malformed_payload: r##"<tool_call>not json</tool_call>"##,
        open_sentinel_only: r##"<tool_call>"##,
    }),
};

const SMOLLM2: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<tool_call>",
            close: "</tool_call>",
        },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::permissive())
        },
    },
    // Small SmolLM2 instruct fine-tunes have not been validated as
    // reliable parallel-call emitters; expose them as serial only.
    supports_parallel_calls: false,
    prepare_messages: protocols::smollm2::prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<tool_call>[{"name":"add","arguments":{"a":1,"b":2}}]</tool_call>"##,
        unknown_tool: r##"<tool_call>[{"name":"missing","arguments":{}}]</tool_call>"##,
        invalid_args: r##"<tool_call>[{"name":"add","arguments":{"a":"x","b":2}}]</tool_call>"##,
        malformed_payload: r##"<tool_call>not json</tool_call>"##,
        open_sentinel_only: r##"<tool_call>"##,
    }),
};

const SMOLLM3: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<tool_call>",
            close: "</tool_call>",
        },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::permissive())
        },
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"##,
        unknown_tool: r##"<tool_call>{"name":"missing","arguments":{}}</tool_call>"##,
        invalid_args: r##"<tool_call>{"name":"add","arguments":{"a":"x","b":2}}</tool_call>"##,
        malformed_payload: r##"<tool_call>not json</tool_call>"##,
        open_sentinel_only: r##"<tool_call>"##,
    }),
};

const LLAMA3: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Custom(protocols::llama3::make_parser),
    // Llama 3's chat template raises if `tool_calls | length != 1`,
    // so the dialect is structurally single-call per turn.
    supports_parallel_calls: false,
    prepare_messages: render::identity_prepare_messages,
    examples: None,
};

const LFM2: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<|tool_call_start|>",
            close: "<|tool_call_end|>",
        },
        // LFM2 emits Pythonic OR JSON inside the sentinels. Try JSON
        // first (cheaper to fail on non-JSON input), fall back to
        // Pythonic. Behavior matches the original
        // `parse_payload_calls` byte-sniff dispatcher.
        codec_factory: || {
            Box::new(super::codecs::MultiCodec::new(vec![
                Box::new(super::codecs::JsonObjectOrArrayCodec::permissive()),
                Box::new(super::codecs::PythonicCallsCodec),
            ]))
        },
    },
    // LFM2 / LFM2.5 emit a list of calls between a single
    // `<|tool_call_start|>` / `<|tool_call_end|>` pair, so parallel
    // calls in one generation are part of the wire format.
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<|tool_call_start|>[add(a=1, b=2)]<|tool_call_end|>"##,
        unknown_tool: r##"<|tool_call_start|>[missing()]<|tool_call_end|>"##,
        invalid_args: r##"<|tool_call_start|>[add(a="x", b=2)]<|tool_call_end|>"##,
        malformed_payload: r##"<|tool_call_start|>not anything{<|tool_call_end|>"##,
        open_sentinel_only: r##"<|tool_call_start|>"##,
    }),
};

const QWEN3_5: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<tool_call>",
            close: "</tool_call>",
        },
        codec_factory: || Box::new(super::codecs::XmlFunctionCodec),
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>"##,
        unknown_tool: r##"<tool_call><function=missing></function></tool_call>"##,
        invalid_args: r##"<tool_call><function=add><parameter=a>"x"</parameter><parameter=b>2</parameter></function></tool_call>"##,
        malformed_payload: r##"<tool_call>not xml</tool_call>"##,
        open_sentinel_only: r##"<tool_call>"##,
    }),
};

const OLMO3: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<function_calls>",
            close: "</function_calls>",
        },
        codec_factory: || Box::new(super::codecs::PythonicCallsCodec),
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<function_calls>[add(a=1, b=2)]</function_calls>"##,
        unknown_tool: r##"<function_calls>[missing()]</function_calls>"##,
        invalid_args: r##"<function_calls>[add(a="x", b=2)]</function_calls>"##,
        malformed_payload: r##"<function_calls>not pythonic{</function_calls>"##,
        open_sentinel_only: r##"<function_calls>"##,
    }),
};

const NEMOTRON: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Pair {
            open: "<tool_call>",
            close: "</tool_call>",
        },
        codec_factory: || Box::new(super::codecs::XmlFunctionCodec),
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>"##,
        unknown_tool: r##"<tool_call><function=missing></function></tool_call>"##,
        invalid_args: r##"<tool_call><function=add><parameter=a>"x"</parameter><parameter=b>2</parameter></function></tool_call>"##,
        malformed_payload: r##"<tool_call>not xml</tool_call>"##,
        open_sentinel_only: r##"<tool_call>"##,
    }),
};

const GRANITE: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Prefix {
            open: "<|tool_call|>",
        },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::strict())
        },
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<|tool_call|>[{"name":"add","arguments":{"a":1,"b":2}}]"##,
        unknown_tool: r##"<|tool_call|>[{"name":"missing","arguments":{}}]"##,
        invalid_args: r##"<|tool_call|>[{"name":"add","arguments":{"a":"x","b":2}}]"##,
        malformed_payload: r##"<|tool_call|>not json at all"##,
        open_sentinel_only: r##"<|tool_call|>"##,
    }),
};

const MISTRAL3: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Prefix {
            open: "[TOOL_CALLS]",
        },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::strict())
        },
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"[TOOL_CALLS][{"name":"add","arguments":{"a":1,"b":2}}]"##,
        unknown_tool: r##"[TOOL_CALLS][{"name":"missing","arguments":{}}]"##,
        invalid_args: r##"[TOOL_CALLS][{"name":"add","arguments":{"a":"x","b":2}}]"##,
        malformed_payload: r##"[TOOL_CALLS]not json"##,
        open_sentinel_only: r##"[TOOL_CALLS]"##,
    }),
};

const PHI4: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        sentinel: SentinelKind::Prefix { open: "functools" },
        codec_factory: || {
            Box::new(super::codecs::JsonObjectOrArrayCodec::strict())
        },
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"functools[{"name":"add","arguments":{"a":1,"b":2}}]"##,
        unknown_tool: r##"functools[{"name":"missing","arguments":{}}]"##,
        invalid_args: r##"functools[{"name":"add","arguments":{"a":"x","b":2}}]"##,
        malformed_payload: r##"functoolsnot json"##,
        open_sentinel_only: r##"functools"##,
    }),
};

const GPT_OSS: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Custom(protocols::gpt_oss::make_parser),
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: None,
};

const GEMMA4: ToolCallProtocol = ToolCallProtocol {
    render_tools: render::openai_tool_envelope,
    parser: ParserShape::Engine {
        // Asymmetric sentinels: open is `<|tool_call>` (uses `|>`),
        // close is `<tool_call|>` (uses `<|`). Intentional in the
        // Gemma 4 wire format — distinct token IDs, not a typo.
        sentinel: SentinelKind::Pair {
            open: "<|tool_call>",
            close: "<tool_call|>",
        },
        codec_factory: || Box::new(super::codecs::Gemma4Codec),
    },
    supports_parallel_calls: true,
    prepare_messages: render::identity_prepare_messages,
    examples: Some(ProtocolExamples {
        valid_call_add_1_2: r##"<|tool_call>call:add{a:1,b:2}<tool_call|>"##,
        unknown_tool: r##"<|tool_call>call:missing{}<tool_call|>"##,
        invalid_args: r##"<|tool_call>call:add{a:<|"|>x<|"|>,b:2}<tool_call|>"##,
        malformed_payload: r##"<|tool_call>this isn't valid<tool_call|>"##,
        open_sentinel_only: r##"<|tool_call>"##,
    }),
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
        // Gemma 4 — separate `Gemma4ForConditionalGeneration` arch.
        "Gemma4ForConditionalGeneration" => Some(&GEMMA4),
        // Qwen3.5 family — text and conditional-generation variants
        // share the same XML-in-`<tool_call>` dialect.
        "Qwen3_5ForCausalLM"
        | "Qwen3_5MoeForCausalLM"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration" => Some(&QWEN3_5),
        // LFM2 (text-only) and LFM2-VL share the same tool-call wire
        // format; both use the same protocol.
        "Lfm2ForCausalLM" | "Lfm2VlForConditionalGeneration" => Some(&LFM2),
        // OLMo 3 — Pythonic in `<function_calls>...</function_calls>`.
        "Olmo3ForCausalLM" => Some(&OLMO3),
        // Nemotron / Nemotron-H — Hermes-style JSON in `<tool_call>`.
        "NemotronForCausalLM" | "NemotronHForCausalLM" => Some(&NEMOTRON),
        // IBM Granite 3.x — JSON list after `<|tool_call|>`.
        "GraniteForCausalLM" | "GraniteMoeForCausalLM" => Some(&GRANITE),
        // Mistral / Ministral with `[TOOL_CALLS]` prefix sentinel.
        "MistralForCausalLM" | "Mistral3ForCausalLM" | "Ministral3ForCausalLM" => {
            Some(&MISTRAL3)
        }
        // Phi-3 / Phi-4-mini — `functools[...]` prefix sentinel.
        "Phi3ForCausalLM" | "Phi4ForCausalLM" => Some(&PHI4),
        // gpt-oss harmony channels.
        "GptOssForCausalLM" => Some(&GPT_OSS),
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
    fn lfm2_architectures_resolve_to_protocol() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("Lfm2ForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("Lfm2VlForConditionalGeneration", &cfg).is_some());
    }

    #[test]
    fn extended_architectures_resolve_to_protocol() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("Qwen3_5ForConditionalGeneration", &cfg).is_some());
        assert!(tool_protocol_for("Olmo3ForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("NemotronHForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("GraniteForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("MistralForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("Phi3ForCausalLM", &cfg).is_some());
        assert!(tool_protocol_for("GptOssForCausalLM", &cfg).is_some());
    }

    #[test]
    fn gemma4_architecture_resolves_to_protocol() {
        let cfg = JsonValue::Null;
        assert!(tool_protocol_for("Gemma4ForConditionalGeneration", &cfg).is_some());
    }

    #[test]
    fn unknown_architecture_returns_none() {
        let cfg = JsonValue::Null;
        // Gemma3 / DeepseekV3 — no tool dialect registered yet.
        assert!(tool_protocol_for("Gemma3ForCausalLM", &cfg).is_none());
        assert!(tool_protocol_for("DeepseekV3ForCausalLM", &cfg).is_none());
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
        let mut parser = proto.make_parser(dir);
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
        let mut parser = proto.make_parser(dir);
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
