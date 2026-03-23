use crate::helpers::WeightPostProcess;
use crate::utils::get_model;
use crate::{LLMError, Result};
use bincode::config;
use catgrad::category::core::{Dtype, Shape};
use catgrad::category::lang::TypedTerm;
use catgrad::prelude::{DynModule, Path};
use std::ops::Deref;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgramSpec {
    pub typed_term: TypedTerm,
    pub module_path: Path,
    pub empty_state_type: Vec<(Dtype, Shape)>,
    pub max_sequence_length: usize,
    pub weight_post_process: WeightPostProcess,
}

impl ProgramSpec {
    pub fn from_typed_term(
        typed_term: TypedTerm,
        module_path: Path,
        empty_state_type: Vec<(Dtype, Shape)>,
        max_sequence_length: usize,
        weight_post_process: WeightPostProcess,
    ) -> Self {
        Self {
            typed_term,
            module_path,
            empty_state_type,
            max_sequence_length,
            weight_post_process,
        }
    }

    pub fn from_module(
        module: &dyn DynModule,
        module_path: Path,
        empty_state_type: Vec<(Dtype, Shape)>,
        max_sequence_length: usize,
        weight_post_process: WeightPostProcess,
    ) -> Result<Self> {
        let typed_term = module.term().ok_or_else(|| {
            LLMError::InvalidProgram("failed to build typed term from module".to_string())
        })?;
        Ok(Self::from_typed_term(
            typed_term,
            module_path,
            empty_state_type,
            max_sequence_length,
            weight_post_process,
        ))
    }

    pub fn text_from_config(
        config_json: &serde_json::Value,
        max_sequence_length: usize,
    ) -> Result<Self> {
        let model = get_model(config_json, max_sequence_length)?;
        let module_path = model.path();
        let empty_state_type = model.empty_state_type();
        let weight_post_process = model.weight_post_process();
        Self::from_module(
            model.as_ref(),
            module_path,
            empty_state_type,
            max_sequence_length,
            weight_post_process,
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, config::standard())
            .map_err(|err| LLMError::InvalidProgram(format!("failed to encode program: {err}")))
    }

    pub fn id(&self) -> Result<String> {
        let bytes = self.canonical_bytes()?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Clone, Debug)]
pub struct Program(Arc<ProgramInner>);

#[derive(Debug)]
struct ProgramInner {
    spec: ProgramSpec,
    id: Arc<str>,
    canonical_bytes: Arc<[u8]>,
}

impl Program {
    pub fn from_spec(spec: ProgramSpec) -> Result<Self> {
        let canonical_bytes: Arc<[u8]> = spec.canonical_bytes()?.into();
        let id = Arc::<str>::from(blake3::hash(&canonical_bytes).to_hex().to_string());
        Ok(Self(Arc::new(ProgramInner {
            spec,
            id,
            canonical_bytes,
        })))
    }

    pub fn spec(&self) -> &ProgramSpec {
        &self.0.spec
    }

    pub fn id(&self) -> &str {
        &self.0.id
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.0.canonical_bytes
    }
}

impl TryFrom<&[u8]> for Program {
    type Error = LLMError;

    fn try_from(bytes: &[u8]) -> std::result::Result<Self, Self::Error> {
        let (spec, consumed) =
            bincode::serde::decode_from_slice(bytes, config::standard()).map_err(|err| {
                LLMError::InvalidProgram(format!("failed to decode program: {err}"))
            })?;
        if consumed != bytes.len() {
            return Err(LLMError::InvalidProgram(format!(
                "program payload had {} trailing bytes",
                bytes.len().saturating_sub(consumed)
            )));
        }
        Self::from_spec(spec)
    }
}

impl Deref for Program {
    type Target = ProgramSpec;

    fn deref(&self) -> &Self::Target {
        self.spec()
    }
}

#[cfg(test)]
mod tests {
    use super::{Program, ProgramSpec};
    use crate::helpers::WeightPostProcess;
    use catgrad::category::core::{Dtype, Shape};
    use catgrad::category::lang::TypedTerm;
    use catgrad::path::Path;

    #[test]
    fn canonical_program_encodes_and_hashes_spec() {
        let spec = ProgramSpec::from_typed_term(
            TypedTerm {
                term: catgrad::category::lang::Term::empty(),
                source_type: vec![],
                target_type: vec![],
            },
            Path::empty(),
            vec![(Dtype::F32, Shape(vec![1, 2, 3]))],
            42,
            WeightPostProcess::None,
        );
        let expected_id = spec.id().unwrap();
        let expected_bytes = spec.canonical_bytes().unwrap();

        let program = Program::from_spec(spec).unwrap();
        assert_eq!(program.id(), expected_id);
        assert_eq!(program.canonical_bytes(), expected_bytes.as_slice());
    }

    #[test]
    fn canonical_program_round_trips_binary_encoding() {
        let spec = ProgramSpec::from_typed_term(
            TypedTerm {
                term: catgrad::category::lang::Term::empty(),
                source_type: vec![],
                target_type: vec![],
            },
            Path::empty(),
            vec![(Dtype::F32, Shape(vec![4, 5]))],
            7,
            WeightPostProcess::None,
        );

        let program = Program::from_spec(spec).unwrap();
        let reparsed = Program::try_from(program.canonical_bytes()).unwrap();

        assert_eq!(reparsed.id(), program.id());
        assert_eq!(reparsed.canonical_bytes(), program.canonical_bytes());
    }
}
