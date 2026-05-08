//! Content-addressed catgrad artifact primitives.
//!
//! `catnix` gives catgrad-shaped objects Nix-like input and output addresses.
//! It deliberately does not know about Hellas receipts, producer signatures,
//! settlement, prices, or evidence.

use std::marker::PhantomData;

const SOURCE_INPUT_SCHEMA: &str = "catnix.source.input.v1";
const SOURCE_OUTPUT_SCHEMA: &str = "catnix.source.output.v1";
const TEXT_INPUT_SCHEMA: &str = "catnix.text.input.v1";
const TEXT_POLICY_SCHEMA: &str = "catnix.text.policy.v1";
const TEXT_EXECUTION_SCHEMA: &str = "catnix.text.execution.v1";
const TEXT_ARTIFACT_IDENTITY_SCHEMA: &str = "catnix.text.artifact.identity.v1";
const TEXT_ARTIFACT_OUTPUT_SCHEMA: &str = "catnix.text.artifact.output.v1";

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Digest({self})")
    }
}

pub trait Canonical {
    fn encode(&self, encoder: &mut DagCborEncoder);

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = DagCborEncoder::new();
        self.encode(&mut encoder);
        encoder.into_bytes()
    }
}

pub trait InputAddressed: Canonical {
    type Artifact: OutputAddressed;

    fn input_id(&self) -> InputId<Self>
    where
        Self: Sized,
    {
        InputId::from_digest(Digest::from_canonical_bytes(&self.canonical_bytes()))
    }
}

pub trait OutputAddressed: Canonical {
    fn output_id(&self) -> OutputId<Self>
    where
        Self: Sized,
    {
        OutputId::from_digest(Digest::from_canonical_bytes(&self.canonical_bytes()))
    }
}

pub struct InputId<I> {
    digest: Digest,
    _ty: PhantomData<I>,
}

impl<I> InputId<I> {
    pub const fn from_digest(digest: Digest) -> Self {
        Self {
            digest,
            _ty: PhantomData,
        }
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self::from_digest(Digest::from_bytes(bytes))
    }

    pub const fn digest(&self) -> Digest {
        self.digest
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.digest.as_bytes()
    }
}

impl<I> Clone for InputId<I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for InputId<I> {}

impl<I> PartialEq for InputId<I> {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl<I> Eq for InputId<I> {}

impl<I> std::hash::Hash for InputId<I> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl<I> std::fmt::Display for InputId<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.digest.fmt(f)
    }
}

impl<I> std::fmt::Debug for InputId<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InputId({})", self.digest)
    }
}

pub struct OutputId<O> {
    digest: Digest,
    _ty: PhantomData<O>,
}

impl<O> OutputId<O> {
    pub const fn from_digest(digest: Digest) -> Self {
        Self {
            digest,
            _ty: PhantomData,
        }
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self::from_digest(Digest::from_bytes(bytes))
    }

    pub const fn digest(&self) -> Digest {
        self.digest
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.digest.as_bytes()
    }
}

impl<O> Clone for OutputId<O> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<O> Copy for OutputId<O> {}

impl<O> PartialEq for OutputId<O> {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl<O> Eq for OutputId<O> {}

impl<O> std::hash::Hash for OutputId<O> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl<O> std::fmt::Display for OutputId<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.digest.fmt(f)
    }
}

impl<O> std::fmt::Debug for OutputId<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OutputId({})", self.digest)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceRef<I: InputAddressed> {
    Input(InputId<I>),
    Output(OutputId<I::Artifact>),
}

impl<I: InputAddressed> SourceRef<I> {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        match self {
            Self::Input(id) => {
                encoder.array(2);
                encoder.str(SOURCE_INPUT_SCHEMA);
                encoder.bytes(id.as_bytes());
            }
            Self::Output(id) => {
                encoder.array(2);
                encoder.str(SOURCE_OUTPUT_SCHEMA);
                encoder.bytes(id.as_bytes());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundTerm;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateBundle;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextInput {
    tokens: OutputId<Tensor>,
}

impl TextInput {
    pub const fn new(tokens: OutputId<Tensor>) -> Self {
        Self { tokens }
    }

    pub const fn tokens(&self) -> OutputId<Tensor> {
        self.tokens
    }
}

impl Canonical for TextInput {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(2);
        encoder.str(TEXT_INPUT_SCHEMA);
        encoder.bytes(self.tokens.as_bytes());
    }
}

impl OutputAddressed for TextInput {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPolicy {
    max_new_tokens: u32,
    stop_token_ids: Vec<i32>,
}

impl TextPolicy {
    pub fn new(max_new_tokens: u32, mut stop_token_ids: Vec<i32>) -> Self {
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        Self {
            max_new_tokens,
            stop_token_ids,
        }
    }

    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    pub fn stop_token_ids(&self) -> &[i32] {
        &self.stop_token_ids
    }
}

impl Canonical for TextPolicy {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(3);
        encoder.str(TEXT_POLICY_SCHEMA);
        encoder.u64(self.max_new_tokens as u64);
        encoder.array(self.stop_token_ids.len() as u64);
        for token in &self.stop_token_ids {
            encoder.i64(*token as i64);
        }
    }
}

impl OutputAddressed for TextPolicy {}

pub type TextSource = SourceRef<TextExecution>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextExecution {
    from: TextSource,
    input: OutputId<TextInput>,
    policy: OutputId<TextPolicy>,
}

impl TextExecution {
    pub const fn new(
        from: TextSource,
        input: OutputId<TextInput>,
        policy: OutputId<TextPolicy>,
    ) -> Self {
        Self {
            from,
            input,
            policy,
        }
    }

    pub const fn from(&self) -> &TextSource {
        &self.from
    }

    pub const fn input(&self) -> OutputId<TextInput> {
        self.input
    }

    pub const fn policy(&self) -> OutputId<TextPolicy> {
        self.policy
    }
}

impl Canonical for TextExecution {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(4);
        encoder.str(TEXT_EXECUTION_SCHEMA);
        self.from.encode(encoder);
        encoder.bytes(self.input.as_bytes());
        encoder.bytes(self.policy.as_bytes());
    }
}

impl InputAddressed for TextExecution {
    type Artifact = TextArtifact;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIdentity {
    bound_term: OutputId<BoundTerm>,
}

impl TextIdentity {
    pub const fn new(bound_term: OutputId<BoundTerm>) -> Self {
        Self { bound_term }
    }

    pub const fn bound_term(&self) -> OutputId<BoundTerm> {
        self.bound_term
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextOutput {
    execution: InputId<TextExecution>,
    position: u64,
    state: OutputId<StateBundle>,
    output_tokens: OutputId<Tensor>,
}

impl TextOutput {
    pub const fn new(
        execution: InputId<TextExecution>,
        position: u64,
        state: OutputId<StateBundle>,
        output_tokens: OutputId<Tensor>,
    ) -> Self {
        Self {
            execution,
            position,
            state,
            output_tokens,
        }
    }

    pub const fn execution(&self) -> InputId<TextExecution> {
        self.execution
    }

    pub const fn position(&self) -> u64 {
        self.position
    }

    pub const fn state(&self) -> OutputId<StateBundle> {
        self.state
    }

    pub const fn output_tokens(&self) -> OutputId<Tensor> {
        self.output_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextArtifact {
    Identity(TextIdentity),
    Output(TextOutput),
}

impl Canonical for TextArtifact {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        match self {
            Self::Identity(identity) => {
                encoder.array(2);
                encoder.str(TEXT_ARTIFACT_IDENTITY_SCHEMA);
                encoder.bytes(identity.bound_term.as_bytes());
            }
            Self::Output(output) => {
                encoder.array(5);
                encoder.str(TEXT_ARTIFACT_OUTPUT_SCHEMA);
                encoder.bytes(output.execution.as_bytes());
                encoder.u64(output.position);
                encoder.bytes(output.state.as_bytes());
                encoder.bytes(output.output_tokens.as_bytes());
            }
        }
    }
}

impl OutputAddressed for TextArtifact {}

pub struct DagCborEncoder {
    bytes: Vec<u8>,
}

impl DagCborEncoder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn array(&mut self, len: u64) {
        self.header(4, len);
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.header(2, value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    pub fn str(&mut self, value: &str) {
        self.header(3, value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.header(0, value);
    }

    pub fn i64(&mut self, value: i64) {
        if value >= 0 {
            self.header(0, value as u64);
        } else {
            self.header(1, (-1_i128 - value as i128) as u64);
        }
    }

    fn header(&mut self, major: u8, value: u64) {
        let major = major << 5;
        match value {
            0..=23 => self.bytes.push(major | value as u8),
            24..=0xff => self.bytes.extend_from_slice(&[major | 24, value as u8]),
            0x100..=0xffff => {
                self.bytes.push(major | 25);
                self.bytes.extend_from_slice(&(value as u16).to_be_bytes());
            }
            0x1_0000..=0xffff_ffff => {
                self.bytes.push(major | 26);
                self.bytes.extend_from_slice(&(value as u32).to_be_bytes());
            }
            _ => {
                self.bytes.push(major | 27);
                self.bytes.extend_from_slice(&value.to_be_bytes());
            }
        }
    }
}

impl Default for DagCborEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BoundTerm, InputAddressed, OutputAddressed, OutputId, SourceRef, StateBundle, Tensor,
        TextArtifact, TextExecution, TextIdentity, TextInput, TextOutput, TextPolicy,
    };

    fn output_id<T>(byte: u8) -> OutputId<T> {
        OutputId::from_bytes([byte; 32])
    }

    #[test]
    fn policy_canonicalizes_stop_ids() {
        let a = TextPolicy::new(16, vec![2, 1, 2]);
        let b = TextPolicy::new(16, vec![1, 2]);
        assert_eq!(a.stop_token_ids(), &[1, 2]);
        assert_eq!(a.output_id(), b.output_id());
    }

    #[test]
    fn identity_is_output_addressed_genesis() {
        let identity = TextArtifact::Identity(TextIdentity::new(output_id::<BoundTerm>(7)));
        let input = TextInput::new(output_id::<Tensor>(1)).output_id();
        let policy = TextPolicy::new(4, vec![]).output_id();
        let execution = TextExecution::new(SourceRef::Output(identity.output_id()), input, policy);

        assert_ne!(
            execution.input_id().as_bytes(),
            identity.output_id().as_bytes()
        );
    }

    #[test]
    fn execution_input_id_changes_when_source_changes() {
        let identity = TextArtifact::Identity(TextIdentity::new(output_id::<BoundTerm>(7)));
        let input = TextInput::new(output_id::<Tensor>(1)).output_id();
        let policy = TextPolicy::new(4, vec![]).output_id();
        let first = TextExecution::new(SourceRef::Output(identity.output_id()), input, policy);
        let second = TextExecution::new(SourceRef::Input(first.input_id()), input, policy);

        assert_ne!(first.input_id(), second.input_id());
    }

    #[test]
    fn output_artifact_id_changes_when_tokens_change() {
        let execution = TextExecution::new(
            SourceRef::Output(
                TextArtifact::Identity(TextIdentity::new(output_id::<BoundTerm>(7))).output_id(),
            ),
            TextInput::new(output_id::<Tensor>(1)).output_id(),
            TextPolicy::new(4, vec![]).output_id(),
        )
        .input_id();
        let a = TextArtifact::Output(TextOutput::new(
            execution,
            5,
            output_id::<StateBundle>(8),
            output_id::<Tensor>(1),
        ));
        let b = TextArtifact::Output(TextOutput::new(
            execution,
            5,
            output_id::<StateBundle>(8),
            output_id::<Tensor>(2),
        ));

        assert_ne!(a.output_id(), b.output_id());
    }
}
