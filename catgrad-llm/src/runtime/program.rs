use crate::helpers::WeightPostProcess;
use crate::utils::get_model;
use crate::{LLMError, Result};
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

    pub fn normalized_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(LLMError::from)
    }

    pub fn id(&self) -> Result<String> {
        let bytes = self.normalized_json()?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Clone, Debug)]
pub struct Program(Arc<ProgramInner>);

#[derive(Debug)]
struct ProgramInner {
    spec: ProgramSpec,
    id: Arc<str>,
    normalized_json: Arc<[u8]>,
}

impl Program {
    pub fn from_spec(spec: ProgramSpec) -> Result<Self> {
        let normalized_json: Arc<[u8]> = spec.normalized_json()?.into();
        let id = Arc::<str>::from(blake3::hash(&normalized_json).to_hex().to_string());
        Ok(Self(Arc::new(ProgramInner {
            spec,
            id,
            normalized_json,
        })))
    }

    pub fn parse_json(bytes: &[u8]) -> Result<Self> {
        let spec = serde_json::from_slice(bytes)?;
        Self::from_spec(spec)
    }

    pub fn spec(&self) -> &ProgramSpec {
        &self.0.spec
    }

    pub fn id(&self) -> &str {
        &self.0.id
    }

    pub fn normalized_json(&self) -> &[u8] {
        &self.0.normalized_json
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
    fn canonical_program_normalizes_and_hashes_spec() {
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
        let expected_json = spec.normalized_json().unwrap();

        let program = Program::from_spec(spec).unwrap();
        assert_eq!(program.id(), expected_id);
        assert_eq!(program.normalized_json(), expected_json.as_slice());
    }
}
