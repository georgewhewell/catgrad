use super::Snapshot;
use crate::{LLMError, Result};
use catgrad::category::lang::TypedTerm;
use catgrad::interpreter::{self, Interpreter};
use catgrad::prelude::Shape;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Session<B: interpreter::Backend> {
    runtime_id: u64,
    program_id: Arc<str>,
    typed_term: Arc<TypedTerm>,
    interpreter: Arc<Interpreter<B>>,
    state_arity: usize,
    state: Vec<interpreter::Value<B>>,
}

impl<B: interpreter::Backend> Session<B> {
    pub(crate) fn from_bound(
        runtime_id: u64,
        program_id: Arc<str>,
        typed_term: Arc<TypedTerm>,
        interpreter: Arc<Interpreter<B>>,
        state_arity: usize,
        snapshot: Snapshot<B>,
    ) -> Result<Self> {
        if snapshot.runtime_id() != runtime_id || snapshot.program_id() != program_id.as_ref() {
            return Err(LLMError::IncompatibleSnapshot);
        }
        if snapshot.state().len() != state_arity {
            return Err(LLMError::InvalidProgram(format!(
                "snapshot state arity {} did not match expected {state_arity}",
                snapshot.state().len()
            )));
        }

        Ok(Self {
            runtime_id,
            program_id,
            typed_term,
            interpreter,
            state_arity,
            state: snapshot.into_state(),
        })
    }

    pub fn snapshot(&self) -> Snapshot<B> {
        Snapshot::new(self.runtime_id, self.program_id.clone(), self.state.clone())
    }

    pub fn into_snapshot(self) -> Snapshot<B> {
        Snapshot::new(self.runtime_id, self.program_id, self.state)
    }

    pub fn run_raw(
        &mut self,
        mut inputs: Vec<interpreter::Value<B>>,
    ) -> Result<Vec<interpreter::Value<B>>> {
        inputs.extend(self.state.iter().cloned());

        let mut results = self
            .interpreter
            .run(self.typed_term.term.clone(), inputs)
            .map_err(|err| LLMError::ExecutionError(err.to_string()))?;

        if results.len() < self.state_arity {
            return Err(LLMError::UnexpectedProgramOutput(format!(
                "program returned {} results for state arity {}",
                results.len(),
                self.state_arity
            )));
        }

        let state_index = results.len() - self.state_arity;
        self.state = results.split_off(state_index);
        Ok(results)
    }

    pub fn step_text(&mut self, input_tokens: &[u32]) -> Result<u32> {
        let input_tensor = interpreter::tensor(
            &self.interpreter.backend,
            Shape(vec![1, input_tokens.len()]),
            input_tokens.to_vec(),
        )
        .map_err(|err| LLMError::ExecutionError(format!("input tensor error: {err:?}")))?;

        let mut outputs = self.run_raw(vec![input_tensor])?;
        if outputs.len() != 1 {
            return Err(LLMError::UnexpectedProgramOutput(format!(
                "text program returned {} non-state outputs",
                outputs.len()
            )));
        }

        let output = outputs.remove(0);
        match output {
            interpreter::Value::Tensor(arr) => match self.interpreter.backend.to_vec(arr) {
                interpreter::TaggedVec::U32(values) => values.last().copied().ok_or_else(|| {
                    LLMError::UnexpectedProgramOutput(
                        "token output tensor was empty".to_string(),
                    )
                }),
                interpreter::TaggedVec::F32(_) => Err(LLMError::UnexpectedProgramOutput(
                    "text program returned f32 tensor output".to_string(),
                )),
            },
            _ => Err(LLMError::UnexpectedProgramOutput(
                "text program returned non-tensor output".to_string(),
            )),
        }
    }
}
