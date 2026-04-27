//! SmolLM2-Instruct tool-call protocol.
//!
//! # Wire format
//!
//! Same Hermes-style sentinel-wrapped JSON as Qwen3:
//!
//! ```text
//! preamble text<tool_call>{"name": "x", "arguments": {...}}</tool_call>
//! more text<tool_call>{"name": "y", "arguments": {...}}</tool_call>
//! ```
//!
//! # Why a separate protocol from Qwen3
//!
//! The wire-level parser is identical (both delegate to
//! [`super::json_sentinel`]), but SmolLM2's chat template
//! (`HuggingFaceTB/SmolLM2-*-Instruct/tokenizer_config.json`) is the
//! plain ChatML loop:
//!
//! ```jinja
//! {% for message in messages %}
//!   {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
//! {% endfor %}
//! ```
//!
//! It does not iterate `tools` and does not inject a tool-format
//! preamble. Out of the box the model has no signal that tools are
//! available or what format to emit. For tool calls to work, the
//! protocol prepends a system message describing the available tools
//! and instructing the model to emit `<tool_call>...</tool_call>`
//! blocks. That is what [`prepare_messages`] does.
//!
//! # Architecture detection
//!
//! SmolLM2 reports itself as `LlamaForCausalLM` in `config.json` —
//! identical to vanilla Llama, Mistral, and SmolLM3. The disambiguator
//! is the tokenizer's `bos_token`: SmolLM2 uses `<|im_start|>` (ChatML),
//! while vanilla Llama / Mistral use `<s>`. See
//! [`super::super::protocol::tool_protocol_for`] for the dispatch.

use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::runtime::chat::codecs::JsonObjectOrArrayCodec;
use crate::runtime::chat::sentinel_engine::SentinelEngine;
use crate::runtime::chat::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use crate::types;


const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Construct a SmolLM2 parser bound to the given tool directory.
///
/// The wire format is the Hermes-style `<tool_call>{...}</tool_call>`
/// (or `<tool_call>[{...}, ...]</tool_call>` for SmolLM2's array form).
/// Built on the generic [`SentinelEngine`] with the Hermes-permissive
/// JSON codec, which (a) treats `<tool_call>[]</tool_call>` as zero
/// calls — SmolLM2's documented "no tool needed" reply — and (b) peels
/// `{"type":"function","function":{...}}` spec-shape echoes back into
/// the canonical call shape.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(SentinelEngine::new_pair(
        directory,
        Box::new(JsonObjectOrArrayCodec::permissive()),
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
    ))
}


/// Inject a system message describing the available tools and the
/// expected `<tool_call>...</tool_call>` wire format, then return the
/// modified message list.
///
/// If the caller already supplied a `system` message, the tool preamble
/// is appended to it (separated by a blank line) so caller-provided
/// instructions still flow through. Otherwise a fresh system message is
/// prepended at the front.
///
/// Tools are serialized as a single JSON array of OpenAI-style
/// `{"type":"function","function":{"name","description","parameters"}}`
/// envelopes — the same shape Qwen3's template emits, so the model's
/// pre-training exposure to that pattern is reused.
pub fn prepare_messages(
    specs: &[ToolSpec],
    mut messages: Vec<types::Message>,
) -> Vec<types::Message> {
    if specs.is_empty() {
        return messages;
    }
    let preamble = render_tool_system_prompt(specs);

    // If the first message is already a system message, append the
    // tool preamble to it so caller intent is preserved. Otherwise,
    // prepend a fresh system message.
    if let Some(first) = messages.first_mut() {
        if let types::Message::OpenAI(openai) = first {
            if openai.role == "system" {
                let existing = openai
                    .content
                    .as_ref()
                    .map(|c| match c {
                        types::openai::MessageContent::Text(t) => t.clone(),
                        types::openai::MessageContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                types::openai::ContentPart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    })
                    .unwrap_or_default();
                let combined = if existing.is_empty() {
                    preamble
                } else {
                    format!("{existing}\n\n{preamble}")
                };
                openai.content = Some(types::openai::MessageContent::Text(combined));
                return messages;
            }
        }
    }

    let system = types::Message::OpenAI(Box::new(types::openai::ChatMessage::system(preamble)));
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(system);
    out.extend(messages);
    out
}

/// Renders the system prompt SmolLM2-Instruct expects when tools are
/// bound. Mirrors HuggingFaceTB's
/// [`instructions_function_calling.md`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/instructions_function_calling.md)
/// verbatim — that file is the only published format the
/// model was actually fine-tuned to follow, so deviating measurably
/// degrades quality. Notable shape constraints:
///
/// - Tools are serialized as plain JSON (not the OpenAI
///   `{"type":"function", ...}` envelope) and inlined inside
///   `<tools>...</tools>` tags.
/// - The reply must be a single `<tool_call>[...]</tool_call>` block
///   wrapping an array, even for a single call. Empty array
///   (`<tool_call>[]</tool_call>`) is the model's "no call needed"
///   signal — handled in the parser as zero call events.
fn render_tool_system_prompt(specs: &[ToolSpec]) -> String {
    let tools_array = JsonValue::Array(specs.iter().map(tool_spec_as_json).collect());
    let tools_json = serde_json::to_string(&tools_array).unwrap_or_default();
    format!(
        "You are an expert in composing functions. You are given a question and a set of possible functions. \n\
         Based on the question, you will need to make one or more function/tool calls to achieve the purpose. \n\
         If none of the functions can be used, point it out and refuse to answer. \n\
         If the given question lacks the parameters required by the function, also point it out.\n\n\
         You have access to the following tools:\n\
         <tools>{tools_json}</tools>\n\n\
         The output MUST strictly adhere to the following format, and NO other text MUST be included.\n\
         The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.\n\
         <tool_call>[\n\
         {{\"name\": \"func_name1\", \"arguments\": {{\"argument1\": \"value1\", \"argument2\": \"value2\"}}}},\n\
         ... (more tool calls as required)\n\
         ]</tool_call>"
    )
}

fn tool_spec_as_json(spec: &ToolSpec) -> JsonValue {
    // The official SmolLM2 example feeds tools to the prompt via
    // `transformers.utils.get_json_schema`, which produces a flat
    // `{"name", "description", "parameters"}` shape (no
    // `{"type": "function", ...}` wrapper). We mirror that shape
    // exactly so the model's pre-training distribution lines up.
    let mut object = serde_json::Map::new();
    object.insert("name".into(), JsonValue::String(spec.name.clone()));
    if let Some(description) = &spec.description {
        object.insert(
            "description".into(),
            JsonValue::String(description.clone()),
        );
    }
    object.insert("parameters".into(), spec.parameters.clone());
    JsonValue::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{
        DecodeEvent, IncrementalToolCallParser, StopReason, ToolDirectory, ToolSpec,
    };
    use serde_json::json;

    /// Universal sentinel-engine scenarios via the shared harness.
    /// SmolLM2's wire format is the same Hermes pair as Qwen3 (delegates
    /// the same permissive JSON codec); the SmolLM2-specific tests
    /// below cover prepare_messages and the empty-array zero-calls
    /// signal.
    #[test]
    fn passes_universal_scenarios() {
        use crate::runtime::chat::protocol_test_kit::{ProtocolTestFixture, directory_with_add};
        ProtocolTestFixture {
            make_parser: Box::new(make_parser),
            directory: directory_with_add(),
            valid_call_add_1_2: r##"<tool_call>[{"name":"add","arguments":{"a":1,"b":2}}]</tool_call>"##,
            unknown_tool_call: r##"<tool_call>[{"name":"missing","arguments":{}}]</tool_call>"##,
            invalid_args_call: r##"<tool_call>[{"name":"add","arguments":{"a":"x","b":2}}]</tool_call>"##,
            malformed_payload: r##"<tool_call>not json</tool_call>"##,
            open_sentinel_only: Some(r##"<tool_call>"##),
        }
        .run_universal_scenarios();
    }

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

    #[test]
    fn parser_uses_same_wire_format_as_qwen3() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    /// SmolLM2 was tuned to wrap calls in a JSON array even for a
    /// single call (per HF's official `instructions_function_calling.md`).
    /// The parser must accept that shape — the older Hermes "single
    /// object" form continues to work via the same code path.
    #[test]
    fn parser_accepts_array_payload_with_single_call() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<tool_call>[{"name":"add","arguments":{"a":1,"b":2}}]</tool_call>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    /// SmolLM2's "no tool needed" reply is the literal string
    /// `<tool_call>[]</tool_call>`. It must not become a fatal parse
    /// error — operator gets a clean turn with no tool calls.
    #[test]
    fn parser_accepts_empty_array_as_no_call() {
        let mut p = make_parser(directory_with_add());
        let events = run(&mut *p, &["<tool_call>[]</tool_call>"]);
        // No ToolCallStart events; the parser passes through to Stop.
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::ToolCallStart { .. })));
        assert!(matches!(events.last(), Some(DecodeEvent::Stop { .. })));
    }

    #[test]
    fn parser_handles_multiple_calls_in_one_array() {
        let mul = ToolSpec::new(
            "mul",
            None,
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
            }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![add_tool(), mul]).unwrap());
        let mut p = make_parser(dir);
        let events = run(
            &mut *p,
            &[r#"<tool_call>[{"name":"add","arguments":{"a":1,"b":2}},{"name":"mul","arguments":{"a":3,"b":4}}]</tool_call>"#],
        );
        // Two calls × 3 events each + 1 Stop.
        assert_eq!(events.len(), 7);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { index: 0, name } if name == "add"));
        assert!(matches!(&events[3], DecodeEvent::ToolCallStart { index: 1, name } if name == "mul"));
    }

    /// Spec-shape echo defence: model emits the OpenAI tool-spec
    /// envelope `{"type":"function","function":{"name":...,
    /// "arguments":...}}` instead of the response shape. Peel and
    /// recover.
    #[test]
    fn parser_peels_openai_function_envelope() {
        let mut p = make_parser(directory_with_add());
        let events = run(
            &mut *p,
            &[r#"<tool_call>[{"type":"function","function":{"name":"add","arguments":{"a":1,"b":2}}}]</tool_call>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { name, .. } if name == "add"));
    }

    #[test]
    fn prepare_messages_prepends_system_when_none_present() {
        let user = types::Message::OpenAI(Box::new(types::openai::ChatMessage::user("hi")));
        let messages = prepare_messages(&[add_tool()], vec![user]);
        assert_eq!(messages.len(), 2);
        let types::Message::OpenAI(first) = &messages[0] else {
            panic!("expected OpenAI message at index 0");
        };
        assert_eq!(first.role, "system");
        let content = match first.content.as_ref().expect("system content present") {
            types::openai::MessageContent::Text(t) => t.clone(),
            _ => panic!("expected text content"),
        };
        assert!(content.contains("<tool_call>"));
        assert!(content.contains("\"add\""));
    }

    #[test]
    fn prepare_messages_appends_to_existing_system() {
        let sys =
            types::Message::OpenAI(Box::new(types::openai::ChatMessage::system("be terse.")));
        let user = types::Message::OpenAI(Box::new(types::openai::ChatMessage::user("hi")));
        let messages = prepare_messages(&[add_tool()], vec![sys, user]);
        assert_eq!(messages.len(), 2, "no new system message inserted");
        let types::Message::OpenAI(first) = &messages[0] else {
            panic!("expected OpenAI message at index 0");
        };
        assert_eq!(first.role, "system");
        let content = match first.content.as_ref().expect("system content present") {
            types::openai::MessageContent::Text(t) => t.clone(),
            _ => panic!("expected text content"),
        };
        assert!(
            content.starts_with("be terse."),
            "caller's system text must come first; got {content:?}"
        );
        assert!(content.contains("<tool_call>"));
    }

    #[test]
    fn prepare_messages_with_no_specs_is_identity() {
        let user = types::Message::OpenAI(Box::new(types::openai::ChatMessage::user("hi")));
        let messages = prepare_messages(&[], vec![user.clone()]);
        assert_eq!(messages.len(), 1);
        assert_eq!(&messages[0], &user);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            "hello world",
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
            r#"prefix <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> suffix"#,
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call><tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>"#,
            "the docs say <tool_call> but it's just text",
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        ]
    }

    proptest! {
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..6,
            split in 0_usize..200,
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &[split]);
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }
    }
}
