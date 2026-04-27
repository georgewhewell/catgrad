//! Per-architecture tool-call protocol implementations.
//!
//! Each submodule provides a `make_parser(&ToolDirectory)` constructor
//! returning a boxed [`IncrementalToolCallParser`](super::IncrementalToolCallParser).
//! Architectures missing here cannot serve tool-enabled chat requests:
//! [`ChatTurn::new`](super::ChatTurn::new) (added in a subsequent
//! patch) consults the [`tool_protocol_for`](super::tool_protocol_for)
//! registry and rejects the turn at construction time.

pub mod gemma4;
pub mod gpt_oss;
pub mod granite;
pub mod json_sentinel;
pub mod lfm2;
pub mod llama3;
pub mod mistral3;
pub mod nemotron;
pub mod olmo3;
pub mod phi4;
pub mod qwen3;
pub mod qwen3_5;
pub mod smollm2;
pub mod smollm3;

#[cfg(test)]
pub(crate) mod test_util;
