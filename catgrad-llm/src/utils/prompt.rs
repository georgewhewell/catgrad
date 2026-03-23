use crate::{LLMError, Result, types};
use chrono::Local;
use minijinja::{Environment, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Map as JsonMap, Value as JsonValue};
use tokenizers::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
enum PromptRequestInner {
    Plain(String),
    Chat {
        message_context: Vec<Value>,
        enable_thinking: bool,
    },
}

#[derive(Debug, Clone)]
pub struct PromptRequest(PromptRequestInner);

/// Tokenized model input and its stop tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub input_ids: Vec<i32>,
    pub stop_token_ids: Vec<i32>,
}

impl PreparedPrompt {
    /// Builds a prepared prompt from token ids and stop ids.
    pub fn new(input_ids: Vec<i32>, stop_token_ids: Vec<i32>) -> Self {
        Self {
            input_ids,
            stop_token_ids,
        }
    }

    /// Tokenizes a raw prompt string into model input ids.
    pub fn from_prompt(
        tokenizer: &Tokenizer,
        prompt: &str,
        stop_token_ids: &[i32],
    ) -> Result<Self> {
        let encoding = tokenizer.encode(prompt, true)?;
        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        Ok(Self::new(input_ids, stop_token_ids.to_vec()))
    }

    /// Renders a normalized prompt request and tokenizes the result.
    pub fn from_request(
        tokenizer: &Tokenizer,
        chat_template: Option<&str>,
        request: &PromptRequest,
        stop_token_ids: &[i32],
    ) -> Result<Self> {
        let prompt = request.render(chat_template)?;
        Self::from_prompt(tokenizer, &prompt, stop_token_ids)
    }
}

impl PromptRequest {
    pub fn plain(prompt: impl Into<String>) -> Self {
        Self(PromptRequestInner::Plain(prompt.into()))
    }

    pub fn single_user(
        prompt: impl Into<String>,
        has_image: bool,
        enable_thinking: bool,
    ) -> Self {
        let prompt = prompt.into();
        let message_context = if has_image {
            let content = vec![
                context!(type => "text", text => prompt),
                context!(type => "image"),
            ];
            vec![Value::from_serialize(
                serde_json::json!({
                    "role": "user",
                    "content": content,
                }),
            )]
        } else {
            vec![Value::from_serialize(
                serde_json::json!({
                    "role": "user",
                    "content": prompt,
                }),
            )]
        };

        Self(PromptRequestInner::Chat {
            message_context,
            enable_thinking,
        })
    }

    pub fn from_messages(messages: &[types::Message], enable_thinking: bool) -> Result<Self> {
        let message_context = messages
            .iter()
            .map(message_to_template_context)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(PromptRequestInner::Chat {
            message_context,
            enable_thinking,
        }))
    }

    pub fn render(&self, chat_template: Option<&str>) -> Result<String> {
        match &self.0 {
            PromptRequestInner::Plain(prompt) => Ok(prompt.clone()),
            PromptRequestInner::Chat {
                message_context,
                enable_thinking,
            } => {
                let chat_template = chat_template.ok_or_else(|| {
                    LLMError::InvalidModelConfig("Missing chat template".to_string())
                })?;
                render_template_messages(chat_template, message_context, *enable_thinking)
            }
        }
    }
}

impl TryFrom<&types::openai::ChatCompletionRequest> for PromptRequest {
    type Error = LLMError;

    fn try_from(value: &types::openai::ChatCompletionRequest) -> Result<Self> {
        let messages = value
            .messages
            .iter()
            .cloned()
            .map(types::Message::from)
            .collect::<Vec<_>>();
        Self::from_messages(&messages, false)
    }
}

impl TryFrom<&types::anthropic::MessageRequest> for PromptRequest {
    type Error = LLMError;

    fn try_from(value: &types::anthropic::MessageRequest) -> Result<Self> {
        let messages = Vec::<types::Message>::from(value);
        Self::from_messages(&messages, false)
    }
}

impl TryFrom<&types::plain::CompletionRequest> for PromptRequest {
    type Error = LLMError;

    fn try_from(value: &types::plain::CompletionRequest) -> Result<Self> {
        Ok(Self::plain(value.prompt.clone()))
    }
}

// Keep a single message as the source of truth and derive template-facing fields eagerly.
pub(crate) fn message_to_template_context(message: &types::Message) -> Result<Value> {
    let mut map = JsonMap::new();

    match message {
        types::Message::OpenAI(msg) => {
            map.insert("role".to_string(), JsonValue::String(msg.role.clone()));
            map.insert(
                "content".to_string(),
                JsonValue::String(match msg.content.as_ref() {
                    Some(content) => openai_content_to_template_string(content)?,
                    None => String::new(),
                }),
            );
        }
        types::Message::Anthropic(msg) => {
            map.insert("role".to_string(), JsonValue::String(msg.role.clone()));
            map.insert(
                "content".to_string(),
                JsonValue::String(anthropic_content_to_template_string(&msg.content)?),
            );
            map.insert(
                "content_blocks".to_string(),
                serde_json::to_value(anthropic_content_to_template_blocks(&msg.content))?,
            );
        }
    }

    Ok(Value::from_serialize(map))
}

fn render_template_messages(
    chat_template: &str,
    messages: &[Value],
    enable_thinking: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("strftime_now", strftime_now);
    env.add_template("chat", chat_template)?;
    let tmpl = env.get_template("chat")?;
    let prompt = tmpl.render(context!(
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => enable_thinking
    ))?;

    Ok(prompt)
}

fn strftime_now(format_str: String) -> String {
    Local::now().format(&format_str).to_string()
}

// HF chat templates generally expect message.content to be a plain string.
// Wire-format content can be structured; flatten it to text for template rendering.
fn openai_content_to_template_string(content: &types::openai::MessageContent) -> Result<String> {
    match content {
        types::openai::MessageContent::Text(text) => Ok(text.clone()),
        types::openai::MessageContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    types::openai::ContentPart::Text { text } => out.push_str(text),
                }
            }
            Ok(out)
        }
    }
}

fn anthropic_content_to_template_string(
    content: &types::anthropic::MessageContent,
) -> Result<String> {
    match content {
        types::anthropic::MessageContent::Text(text) => Ok(text.clone()),
        types::anthropic::MessageContent::Blocks(blocks) => {
            anthropic_blocks_to_template_string(blocks)
        }
    }
}

fn anthropic_content_to_template_blocks(
    content: &types::anthropic::MessageContent,
) -> Vec<types::anthropic::ContentBlock> {
    match content {
        types::anthropic::MessageContent::Text(text) => {
            vec![types::anthropic::ContentBlock::Text { text: text.clone() }]
        }
        types::anthropic::MessageContent::Blocks(blocks) => blocks.clone(),
    }
}

fn anthropic_blocks_to_template_string(
    blocks: &[types::anthropic::ContentBlock],
) -> Result<String> {
    let mut out = String::new();
    for block in blocks {
        match block {
            types::anthropic::ContentBlock::Text { text } => out.push_str(text),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_message_template(message: &types::Message, template: &str) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("test", template).unwrap();
        let context = message_to_template_context(message)?;
        env.get_template("test")
            .unwrap()
            .render(context!(message => context))
            .map_err(Into::into)
    }

    #[test]
    fn openai_parts_flatten_to_template_text() {
        let content = types::openai::MessageContent::Parts(vec![
            types::openai::ContentPart::Text {
                text: "hello".to_string(),
            },
            types::openai::ContentPart::Text {
                text: " world".to_string(),
            },
        ]);
        assert_eq!(
            openai_content_to_template_string(&content).unwrap(),
            "hello world".to_string()
        );
    }

    #[test]
    fn anthropic_blocks_flatten_to_template_text() {
        let content = types::anthropic::MessageContent::Blocks(vec![
            types::anthropic::ContentBlock::Text {
                text: "alpha".to_string(),
            },
            types::anthropic::ContentBlock::Text {
                text: "beta".to_string(),
            },
        ]);
        assert_eq!(
            anthropic_content_to_template_string(&content).unwrap(),
            "alphabeta".to_string()
        );
    }

    #[test]
    fn openai_message_context_exposes_flattened_content() {
        let message =
            types::Message::OpenAI(Box::new(types::openai::ChatMessage::assistant("hello")));
        let rendered = render_message_template(&message, "{{ message.content }}").unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn anthropic_message_context_exposes_structured_blocks_without_breaking_content() {
        let message = types::Message::Anthropic(types::anthropic::AnthropicMessage {
            role: "assistant".to_string(),
            content: types::anthropic::MessageContent::Blocks(vec![
                types::anthropic::ContentBlock::Text {
                    text: "alpha".to_string(),
                },
                types::anthropic::ContentBlock::Text {
                    text: "beta".to_string(),
                },
            ]),
        });

        let rendered = render_message_template(
            &message,
            "{{ message.content }}|{{ message.content_blocks|length }}|{{ message.content_blocks[1].text }}",
        )
        .unwrap();

        assert_eq!(rendered, "alphabeta|2|beta");
    }
}
