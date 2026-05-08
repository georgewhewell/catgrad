//! Content-addressed catgrad artifact primitives.
//!
//! `catnix` gives catgrad-shaped objects Nix-like input and output addresses.
//! It deliberately does not know about Hellas receipts, producer signatures,
//! settlement, prices, or evidence.

use std::marker::PhantomData;

const SOURCE_INPUT_SCHEMA: &str = "catnix.source.input.v1";
const SOURCE_OUTPUT_SCHEMA: &str = "catnix.source.output.v1";
const TOKEN_IDS_SCHEMA: &str = "catnix.token_ids.v1";
const TEXT_POLICY_SCHEMA: &str = "catnix.text.policy.v1";
const TEXT_EXECUTION_SCHEMA: &str = "catnix.text.execution.v1";
const TEXT_STATE_SCHEMA: &str = "catnix.text.state.v1";
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
    pub const fn input(id: InputId<I>) -> Self {
        Self::Input(id)
    }

    pub const fn output(id: OutputId<I::Artifact>) -> Self {
        Self::Output(id)
    }

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

pub type BoundTermId = OutputId<BoundTerm>;
pub type TokenIdsId = OutputId<TokenIds>;
pub type TextPolicyId = OutputId<TextPolicy>;
pub type TextExecutionId = InputId<TextExecution>;
pub type TextArtifactId = OutputId<TextArtifact>;
pub type TextStateId = OutputId<TextState>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(u32);

impl TokenId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for TokenId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for TokenId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<i32> for TokenId {
    type Error = TokenIdError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match u32::try_from(value) {
            Ok(value) => Ok(Self(value)),
            Err(_) => Err(TokenIdError { value }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenIdError {
    value: i32,
}

impl TokenIdError {
    pub const fn value(self) -> i32 {
        self.value
    }
}

impl std::fmt::Display for TokenIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "token id {} is negative", self.value)
    }
}

impl std::error::Error for TokenIdError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIds {
    tokens: Vec<TokenId>,
}

impl TokenIds {
    pub fn new(tokens: impl Into<Vec<TokenId>>) -> Self {
        Self {
            tokens: tokens.into(),
        }
    }

    pub fn from_u32s(tokens: impl IntoIterator<Item = u32>) -> Self {
        Self {
            tokens: tokens.into_iter().map(TokenId::new).collect(),
        }
    }

    pub fn as_slice(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl<const N: usize> From<[u32; N]> for TokenIds {
    fn from(value: [u32; N]) -> Self {
        Self::from_u32s(value)
    }
}

impl From<Vec<u32>> for TokenIds {
    fn from(value: Vec<u32>) -> Self {
        Self::from_u32s(value)
    }
}

impl FromIterator<TokenId> for TokenIds {
    fn from_iter<T: IntoIterator<Item = TokenId>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect::<Vec<_>>())
    }
}

impl FromIterator<u32> for TokenIds {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        Self::from_u32s(iter)
    }
}

impl Canonical for TokenIds {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(2);
        encoder.str(TOKEN_IDS_SCHEMA);
        encoder.array(self.tokens.len() as u64);
        for token in &self.tokens {
            encoder.u64(token.as_u32() as u64);
        }
    }
}

impl OutputAddressed for TokenIds {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPolicy {
    max_new_tokens: u32,
    stop_token_ids: Vec<TokenId>,
}

impl TextPolicy {
    pub fn new(max_new_tokens: u32, stop_token_ids: impl IntoIterator<Item = TokenId>) -> Self {
        let mut stop_token_ids: Vec<_> = stop_token_ids.into_iter().collect();
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        Self {
            max_new_tokens,
            stop_token_ids,
        }
    }

    pub fn from_u32_stop_tokens(
        max_new_tokens: u32,
        stop_token_ids: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self::new(max_new_tokens, stop_token_ids.into_iter().map(TokenId::new))
    }

    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    pub fn stop_token_ids(&self) -> &[TokenId] {
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
            encoder.u64(token.as_u32() as u64);
        }
    }
}

impl OutputAddressed for TextPolicy {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextState {
    tokens: TokenIdsId,
}

impl TextState {
    pub const fn new(tokens: TokenIdsId) -> Self {
        Self { tokens }
    }

    pub const fn tokens(&self) -> TokenIdsId {
        self.tokens
    }
}

impl Canonical for TextState {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(2);
        encoder.str(TEXT_STATE_SCHEMA);
        encoder.bytes(self.tokens.as_bytes());
    }
}

impl OutputAddressed for TextState {}

pub type TextSource = SourceRef<TextExecution>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextExecution {
    from: TextSource,
    prompt_tokens: TokenIdsId,
    policy: TextPolicyId,
}

impl TextExecution {
    pub const fn new(from: TextSource, prompt_tokens: TokenIdsId, policy: TextPolicyId) -> Self {
        Self {
            from,
            prompt_tokens,
            policy,
        }
    }

    pub const fn from(&self) -> &TextSource {
        &self.from
    }

    pub const fn prompt_tokens(&self) -> TokenIdsId {
        self.prompt_tokens
    }

    pub const fn policy(&self) -> TextPolicyId {
        self.policy
    }
}

impl Canonical for TextExecution {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        encoder.array(4);
        encoder.str(TEXT_EXECUTION_SCHEMA);
        self.from.encode(encoder);
        encoder.bytes(self.prompt_tokens.as_bytes());
        encoder.bytes(self.policy.as_bytes());
    }
}

impl InputAddressed for TextExecution {
    type Artifact = TextArtifact;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextOutput {
    execution: TextExecutionId,
    position: u64,
    state: TextStateId,
    generated_tokens: TokenIdsId,
}

impl TextOutput {
    pub const fn new(
        execution: TextExecutionId,
        position: u64,
        state: TextStateId,
        generated_tokens: TokenIdsId,
    ) -> Self {
        Self {
            execution,
            position,
            state,
            generated_tokens,
        }
    }

    pub const fn execution(&self) -> TextExecutionId {
        self.execution
    }

    pub const fn position(&self) -> u64 {
        self.position
    }

    pub const fn state(&self) -> TextStateId {
        self.state
    }

    pub const fn generated_tokens(&self) -> TokenIdsId {
        self.generated_tokens
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextArtifact {
    Identity { bound_term: BoundTermId },
    Output(TextOutput),
}

impl TextArtifact {
    pub const fn identity(bound_term: BoundTermId) -> Self {
        Self::Identity { bound_term }
    }

    pub const fn output(
        execution: TextExecutionId,
        position: u64,
        state: TextStateId,
        generated_tokens: TokenIdsId,
    ) -> Self {
        Self::Output(TextOutput::new(
            execution,
            position,
            state,
            generated_tokens,
        ))
    }
}

impl Canonical for TextArtifact {
    fn encode(&self, encoder: &mut DagCborEncoder) {
        match self {
            Self::Identity { bound_term } => {
                encoder.array(2);
                encoder.str(TEXT_ARTIFACT_IDENTITY_SCHEMA);
                encoder.bytes(bound_term.as_bytes());
            }
            Self::Output(output) => {
                encoder.array(5);
                encoder.str(TEXT_ARTIFACT_OUTPUT_SCHEMA);
                encoder.bytes(output.execution.as_bytes());
                encoder.u64(output.position);
                encoder.bytes(output.state.as_bytes());
                encoder.bytes(output.generated_tokens.as_bytes());
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
        BoundTerm, InputAddressed, OutputAddressed, OutputId, SourceRef, TextArtifact,
        TextExecution, TextPolicy, TextState, TokenId, TokenIds,
    };

    fn output_id<T>(byte: u8) -> OutputId<T> {
        OutputId::from_bytes([byte; 32])
    }

    #[test]
    fn token_ids_are_output_addressed_values() {
        let a = TokenIds::from([1, 2, 3]);
        let b = TokenIds::from([1, 2, 3]);
        let c = TokenIds::from([3, 2, 1]);

        assert_eq!(
            a.as_slice(),
            &[TokenId::new(1), TokenId::new(2), TokenId::new(3)]
        );
        assert_eq!(a.output_id(), b.output_id());
        assert_ne!(a.output_id(), c.output_id());
    }

    #[test]
    fn negative_model_token_ids_are_rejected_at_the_boundary() {
        assert_eq!(TokenId::try_from(7_i32).unwrap(), TokenId::new(7));
        let err = TokenId::try_from(-1_i32).unwrap_err();
        assert_eq!(err.value(), -1);
    }

    #[test]
    fn policy_canonicalizes_stop_ids() {
        let a = TextPolicy::from_u32_stop_tokens(16, [2, 1, 2]);
        let b = TextPolicy::from_u32_stop_tokens(16, [1, 2]);
        assert_eq!(a.stop_token_ids(), &[TokenId::new(1), TokenId::new(2)]);
        assert_eq!(a.output_id(), b.output_id());
    }

    #[test]
    fn text_state_is_output_addressed_by_token_artifact() {
        let a = TextState::new(TokenIds::from([1, 2, 3]).output_id());
        let b = TextState::new(TokenIds::from([1, 2, 3]).output_id());
        let c = TextState::new(TokenIds::from([1, 2, 4]).output_id());

        assert_eq!(a.tokens(), b.tokens());
        assert_eq!(a.output_id(), b.output_id());
        assert_ne!(a.output_id(), c.output_id());
    }

    #[test]
    fn identity_is_output_addressed_genesis() {
        let identity = TextArtifact::identity(output_id::<BoundTerm>(7));
        let prompt_tokens = TokenIds::from([1]).output_id();
        let policy = TextPolicy::from_u32_stop_tokens(4, []).output_id();
        let execution = TextExecution::new(
            SourceRef::output(identity.output_id()),
            prompt_tokens,
            policy,
        );

        assert_ne!(
            execution.input_id().as_bytes(),
            identity.output_id().as_bytes()
        );
    }

    #[test]
    fn execution_input_id_changes_when_source_changes() {
        let identity = TextArtifact::identity(output_id::<BoundTerm>(7));
        let prompt_tokens = TokenIds::from([1]).output_id();
        let policy = TextPolicy::from_u32_stop_tokens(4, []).output_id();
        let first = TextExecution::new(
            SourceRef::output(identity.output_id()),
            prompt_tokens,
            policy,
        );
        let second = TextExecution::new(SourceRef::input(first.input_id()), prompt_tokens, policy);

        assert_ne!(first.input_id(), second.input_id());
    }

    #[test]
    fn output_artifact_id_changes_when_generated_tokens_change() {
        let execution = TextExecution::new(
            SourceRef::output(TextArtifact::identity(output_id::<BoundTerm>(7)).output_id()),
            TokenIds::from([1]).output_id(),
            TextPolicy::from_u32_stop_tokens(4, []).output_id(),
        )
        .input_id();
        let a = TextArtifact::output(
            execution,
            5,
            TextState::new(TokenIds::from([1]).output_id()).output_id(),
            TokenIds::from([1]).output_id(),
        );
        let b = TextArtifact::output(
            execution,
            5,
            TextState::new(TokenIds::from([1]).output_id()).output_id(),
            TokenIds::from([2]).output_id(),
        );

        assert_ne!(a.output_id(), b.output_id());
    }
}
