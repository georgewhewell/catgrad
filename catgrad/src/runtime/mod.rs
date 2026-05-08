//! Execute catgrad typed terms against materialized parameter tensors.
//!
//! A [`BoundTerm`] is the small reusable wrapper around the pattern used
//! by examples, `catgrad-llm`, `chatgrad`, and downstream consumers:
//!
//! 1. Build or receive a [`crate::category::lang::TypedTerm`].
//! 2. Materialize a [`crate::interpreter::Parameters`] map.
//! 3. Bind the term to those parameters and a load-prefix path.
//! 4. Call [`BoundTerm::run`] with the exact ordered input vector the
//!    graph author intended.
//!
//! The runtime layer deliberately does not know which inputs are tokens,
//! state, controls, image embeddings, or anything else. Those semantics
//! belong to the graph author and higher-level packages.
//!
//! ```ignore
//! let bound = BoundTerm::new(typed_term, &backend, &parameters, load_prefix)?;
//! let outputs = bound.run(inputs)?;
//! ```
//!
//! Identity for bound terms and executions lives in the `catnix` crate.
//! catgrad owns execution; catnix owns the address contract.

mod bound;
mod error;

pub use bound::BoundTerm;
pub use error::{Result, RuntimeError};
