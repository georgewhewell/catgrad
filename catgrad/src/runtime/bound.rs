//! A typed term paired with materialized parameter tensors and an
//! [`Interpreter`] ready to run it.

use std::sync::Arc;

use crate::category::lang::TypedTerm;
use crate::interpreter::{self, Backend, Interpreter};
use crate::path::Path;
use crate::prelude::{stdlib, to_load_ops};

use super::error::{Result, RuntimeError};

/// A typed term loaded against a specific set of parameter tensors.
///
/// `BoundTerm` deliberately does not know graph input semantics. The
/// graph author remains responsible for constructing the ordered input
/// vector expected by the term.
#[derive(Debug)]
pub struct BoundTerm<B: Backend> {
    typed_term: Arc<TypedTerm>,
    interpreter: Arc<Interpreter<B>>,
}

impl<B: Backend> Clone for BoundTerm<B> {
    fn clone(&self) -> Self {
        Self {
            typed_term: Arc::clone(&self.typed_term),
            interpreter: Arc::clone(&self.interpreter),
        }
    }
}

impl<B: Backend> BoundTerm<B> {
    /// Bind `typed_term` against the materialized parameters `params`.
    ///
    /// `params` is taken by reference so a single materialized set can
    /// back many bound programs; the tensor map is cloned (shallow —
    /// concrete backend tensors are Arc-wrapped, so this is cheap).
    /// `typed_term` is accepted as either an owned [`TypedTerm`] or an
    /// [`Arc<TypedTerm>`] via [`Into`] — multi-bind callers should pass
    /// `Arc::clone(&shared)` to avoid the deep-clone cost of
    /// [`TypedTerm::clone`].
    pub fn new(
        typed_term: impl Into<Arc<TypedTerm>>,
        backend: &B,
        params: &interpreter::Parameters<B>,
        load_prefix: Path,
    ) -> Result<Self> {
        let typed_term = typed_term.into();
        let mut env = stdlib();
        env.declarations
            .extend(to_load_ops(load_prefix, params.keys()));

        let interpreter = Arc::new(Interpreter::new(backend.clone(), env, params.clone()));
        Ok(Self {
            typed_term,
            interpreter,
        })
    }

    /// The bound typed term.
    pub fn typed_term(&self) -> &TypedTerm {
        &self.typed_term
    }

    /// The underlying interpreter (owns the backend handle and
    /// parameter values). Exposed for callers that need to construct
    /// backend tensors with the same backend handle.
    pub fn interpreter(&self) -> &Interpreter<B> {
        self.interpreter.as_ref()
    }

    /// Run the term with a complete input vector.
    pub fn run(&self, inputs: Vec<interpreter::Value<B>>) -> Result<Vec<interpreter::Value<B>>> {
        self.interpreter
            .run(self.typed_term.term.clone(), inputs)
            .map_err(|err| RuntimeError::ExecutionError(err.to_string()))
    }
}
