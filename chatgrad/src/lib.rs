//! Chat-protocol layer over catgrad-llm: wire-format types, chat-template
//! rendering, tool-call dispatch, and the legacy `run` runner.

pub mod prompt;
pub mod run;
pub mod tool_calls;
pub mod types;

pub use prompt::{
    PreparedPrompt, RenderChatTemplateOptions, render_chat_template, render_chat_template_values,
};
pub use tool_calls::{ToolCall, ToolUseStep};
