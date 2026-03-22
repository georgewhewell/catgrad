use super::{Program, Session, Snapshot};
use catgrad::interpreter::{self, Interpreter};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BoundProgram<B: interpreter::Backend> {
    runtime_id: u64,
    program_id: Arc<str>,
    program: Arc<Program>,
    typed_term: Arc<catgrad::category::lang::TypedTerm>,
    interpreter: Arc<Interpreter<B>>,
}

impl<B: interpreter::Backend> BoundProgram<B> {
    pub(crate) fn new(
        runtime_id: u64,
        program_id: Arc<str>,
        program: Arc<Program>,
        typed_term: Arc<catgrad::category::lang::TypedTerm>,
        interpreter: Arc<Interpreter<B>>,
    ) -> Self {
        Self {
            runtime_id,
            program_id,
            program,
            typed_term,
            interpreter,
        }
    }

    pub fn id(&self) -> &str {
        &self.program_id
    }

    pub fn program(&self) -> &Program {
        self.program.as_ref()
    }

    pub fn empty_snapshot(&self) -> Snapshot<B> {
        let state = self
            .program
            .empty_state_type
            .iter()
            .map(|(dtype, shape)| {
                interpreter::Value::Tensor(
                    self.interpreter.backend.zeros(shape.clone(), dtype.clone()),
                )
            })
            .collect();

        Snapshot::new(self.runtime_id, self.program_id.clone(), state)
    }

    pub fn start(&self, snapshot: Snapshot<B>) -> crate::Result<Session<B>> {
        Session::from_bound(
            self.runtime_id,
            self.program_id.clone(),
            self.typed_term.clone(),
            self.interpreter.clone(),
            self.program.empty_state_type.len(),
            snapshot,
        )
    }
}
