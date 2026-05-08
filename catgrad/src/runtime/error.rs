//! Error type for the runtime layer and the module's `Result` alias.

/// Errors produced when constructing or driving a [`super::BoundTerm`].
#[derive(Debug)]
pub enum RuntimeError {
    ExecutionError(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutionError(msg) => write!(f, "execution error: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub type Result<T, E = RuntimeError> = std::result::Result<T, E>;
