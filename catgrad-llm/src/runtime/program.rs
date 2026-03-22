use crate::helpers::WeightPostProcess;
use crate::utils::get_model;
use crate::{LLMError, Result};
use catgrad::category::core::{Dtype, Shape};
use catgrad::category::lang::TypedTerm;
use catgrad::prelude::{DynModule, Path};

pub const CURRENT_PROGRAM_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramInterface {
    Raw,
    Text,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Program {
    pub version: u32,
    pub interface: ProgramInterface,
    pub typed_term: TypedTerm,
    pub load_prefix: Path,
    pub empty_state_type: Vec<(Dtype, Shape)>,
    pub max_sequence_length: usize,
    pub weight_post_process: WeightPostProcess,
}

impl Program {
    pub fn from_typed_term(
        interface: ProgramInterface,
        typed_term: TypedTerm,
        load_prefix: Path,
        empty_state_type: Vec<(Dtype, Shape)>,
        max_sequence_length: usize,
        weight_post_process: WeightPostProcess,
    ) -> Self {
        Self {
            version: CURRENT_PROGRAM_VERSION,
            interface,
            typed_term,
            load_prefix,
            empty_state_type,
            max_sequence_length,
            weight_post_process,
        }
    }

    pub fn from_module(
        module: &dyn DynModule,
        interface: ProgramInterface,
        load_prefix: Path,
        empty_state_type: Vec<(Dtype, Shape)>,
        max_sequence_length: usize,
        weight_post_process: WeightPostProcess,
    ) -> Result<Self> {
        let typed_term = module.term().ok_or_else(|| {
            LLMError::InvalidProgram("failed to build typed term from module".to_string())
        })?;
        Ok(Self::from_typed_term(
            interface,
            typed_term,
            load_prefix,
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
        let load_prefix = model.path();
        let empty_state_type = model.empty_state_type();
        let weight_post_process = model.weight_post_process();
        Self::from_module(
            model.as_ref(),
            ProgramInterface::Text,
            load_prefix,
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

    pub(crate) fn validate(&self) -> Result<()> {
        if self.version != CURRENT_PROGRAM_VERSION {
            return Err(LLMError::UnsupportedProgramVersion {
                found: self.version,
                expected: CURRENT_PROGRAM_VERSION,
            });
        }
        Ok(())
    }
}
