//! LLM-specific code like tokenization and kv-cache logic which (currently) has to live outside
//! the model graph.
pub mod config;
mod error;
pub mod helpers;
pub mod legacy;
pub mod models;
pub mod runtime;
pub mod types;
pub mod utils;

pub use error::LLMError;
pub use runtime::{BoundProgram, Program, ProgramInterface, Runtime, Session, Snapshot};
pub use utils::{Detokenizer, PreparedPrompt, detokenize_tokens};
pub type Result<T, E = error::LLMError> = std::result::Result<T, E>;
