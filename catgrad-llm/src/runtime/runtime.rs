use super::bound::BoundProgram;
use super::Program;
use crate::helpers::WeightPostProcess;
use crate::utils::post_process_weights;
use crate::{LLMError, Result};
use catgrad::interpreter::{self, Interpreter};
use catgrad::prelude::{stdlib, to_load_ops};
use catgrad::typecheck;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct Runtime<B: interpreter::Backend> {
    id: u64,
    backend: B,
    parameter_values: interpreter::Parameters<B>,
    parameter_types: typecheck::Parameters,
    weight_post_process: WeightPostProcess,
}

impl<B: interpreter::Backend> Runtime<B> {
    pub fn new(
        backend: B,
        program: &Program,
        mut parameter_values: interpreter::Parameters<B>,
        mut parameter_types: typecheck::Parameters,
    ) -> Result<Self> {
        post_process_weights(
            program.weight_post_process,
            &backend,
            &mut parameter_values,
            &mut parameter_types,
        )?;

        Ok(Self {
            id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            backend,
            parameter_values,
            parameter_types,
            weight_post_process: program.weight_post_process,
        })
    }

    pub fn bind(&self, program: Program) -> Result<BoundProgram<B>> {
        if program.weight_post_process != self.weight_post_process {
            return Err(LLMError::IncompatibleRuntime(format!(
                "program expects weight post-process {:?}, runtime was initialized with {:?}",
                program.weight_post_process, self.weight_post_process
            )));
        }

        let mut env = stdlib();
        env.declarations.extend(to_load_ops(
            program.module_path.clone(),
            self.parameter_types.keys(),
        ));

        typecheck::check(&env, &self.parameter_types, program.typed_term.clone()).map_err(
            |err| LLMError::InvalidProgram(format!("program failed typecheck: {err:?}")),
        )?;

        let program_id = program.id()?;
        let typed_term = Arc::new(program.typed_term.clone());
        let program = Arc::new(program);
        let interpreter = Arc::new(Interpreter::new(
            self.backend.clone(),
            env,
            self.parameter_values.clone(),
        ));

        Ok(BoundProgram::new(
            self.id,
            Arc::<str>::from(program_id),
            program,
            typed_term,
            interpreter,
        ))
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }
}
