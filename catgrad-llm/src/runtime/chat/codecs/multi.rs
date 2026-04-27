//! Try-each-codec-in-order dispatcher.
//!
//! Used when one wire format admits multiple payload encodings. The
//! canonical case is LFM2: payloads can be either Pythonic
//! (`[name(k=v, ...)]`) or JSON (`[{...}]`) at the model's discretion.
//! Per the survey, LFM2 sniffed on the first non-whitespace byte then
//! ran the matching codec; this dispatcher achieves the same end result
//! by trying each codec in turn and returning the first non-error
//! outcome — wrong-codec attempts fail fast on parse error and the
//! dispatcher falls through.
//!
//! Order matters: the first codec is the preferred one. Place the more
//! specific / faster-to-fail codec earlier.

use super::super::event::ParserError;
use super::super::sentinel_engine::{CodecOutcome, PayloadCodec};

pub struct MultiCodec {
    codecs: Vec<Box<dyn PayloadCodec>>,
}

impl MultiCodec {
    pub fn new(codecs: Vec<Box<dyn PayloadCodec>>) -> Self {
        debug_assert!(!codecs.is_empty(), "MultiCodec needs at least one codec");
        Self { codecs }
    }
}

impl PayloadCodec for MultiCodec {
    fn parse(&self, payload: &str) -> CodecOutcome {
        let mut last_error: Option<ParserError> = None;
        for codec in &self.codecs {
            match codec.parse(payload) {
                CodecOutcome::Calls(calls) => return CodecOutcome::Calls(calls),
                CodecOutcome::PartialThenError { calls, error } => {
                    return CodecOutcome::PartialThenError { calls, error };
                }
                CodecOutcome::Error(err) => {
                    last_error = Some(err);
                }
            }
        }
        // All codecs failed. Surface the LAST error — the codecs are
        // ordered preferred-first, so the last failure is the most
        // permissive parser's verdict and likely the most informative.
        CodecOutcome::Error(last_error.expect("MultiCodec invariant: at least one codec"))
    }
}
