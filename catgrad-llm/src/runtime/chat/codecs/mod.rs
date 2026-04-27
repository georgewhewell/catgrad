//! Payload codecs for [`SentinelEngine`](super::sentinel_engine::SentinelEngine).
//!
//! Each codec converts the bytes between sentinels into a list of
//! candidate `(name, args)` pairs. Codecs are stateless and don't know
//! about [`ToolDirectory`] — that's the engine's job.
//!
//! The current codec set covers:
//!
//! - [`json::JsonObjectOrArrayCodec`] — the dominant pattern. JSON
//!   object or array of objects, accepts both `arguments` and
//!   `parameters` keys, decodes `arguments` from a JSON string when
//!   the wire form is the OpenAI legacy shape.

pub mod balanced_lexer;
pub mod gemma4_pythonic;
pub mod json;
pub mod multi;
pub mod pythonic;
pub mod xml_function;

pub use gemma4_pythonic::Gemma4Codec;
pub use json::JsonObjectOrArrayCodec;
pub use multi::MultiCodec;
pub use pythonic::PythonicCallsCodec;
pub use xml_function::XmlFunctionCodec;
