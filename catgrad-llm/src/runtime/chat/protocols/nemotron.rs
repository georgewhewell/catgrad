//! Qwen3.5 tool-call protocol — stub. Implementation TODO.
//!
//! See: catgrad-llm/src/helpers/tool_calls.rs::parse_qwen3_5_tool_calls
//! for the legacy reference implementation. This module ports that
//! parser to the streaming runtime/chat protocol contract.

use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::runtime::chat::{
    IncrementalToolCallParser, PassthroughParser, ToolDirectory, ToolSpec,
};

pub fn make_parser(_directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    // STUB — replaced by the full implementation.
    Box::new(PassthroughParser)
}

pub fn render_tools(_specs: &[ToolSpec]) -> JsonValue {
    // STUB
    JsonValue::Array(Vec::new())
}
