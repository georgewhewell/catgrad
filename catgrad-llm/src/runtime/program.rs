use crate::helpers::WeightPostProcess;
use crate::utils::get_model;
use crate::{LLMError, Result};
use catgrad::category::core::{Dtype, Shape};
use catgrad::category::lang::TypedTerm;
use catgrad::prelude::{DynModule, Path};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Program {
    pub typed_term: TypedTerm,
    pub module_path: Path,
    pub empty_state_type: Vec<(Dtype, Shape)>,
    pub max_sequence_length: usize,
    pub weight_post_process: WeightPostProcess,
}

impl Program {
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
