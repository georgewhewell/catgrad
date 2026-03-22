use catgrad::interpreter;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Snapshot<B: interpreter::Backend> {
    runtime_id: u64,
    program_id: Arc<str>,
    state: Vec<interpreter::Value<B>>,
}

impl<B: interpreter::Backend> Snapshot<B> {
    pub(crate) fn new(
        runtime_id: u64,
        program_id: Arc<str>,
        state: Vec<interpreter::Value<B>>,
    ) -> Self {
        Self {
            runtime_id,
            program_id,
            state,
        }
    }

    pub(crate) fn runtime_id(&self) -> u64 {
        self.runtime_id
    }

    pub(crate) fn program_id(&self) -> &str {
        &self.program_id
    }

    pub(crate) fn into_state(self) -> Vec<interpreter::Value<B>> {
        self.state
    }

    pub(crate) fn state(&self) -> &[interpreter::Value<B>] {
        &self.state
    }
}
