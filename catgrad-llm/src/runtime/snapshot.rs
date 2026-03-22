use catgrad::category::core::Dtype;
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

    pub fn logical_bytes(&self) -> usize {
        self.state
            .iter()
            .map(|value| match value {
                interpreter::Value::Tensor(tensor) => {
                    tensor.shape().size().saturating_mul(dtype_size(tensor.dtype()))
                }
                _ => 0,
            })
            .sum()
    }
}

const fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 | Dtype::U32 => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::Snapshot;
    use catgrad::interpreter::backend::shape_only::ShapeOnlyBackend;
    use catgrad::interpreter::{self, Backend};
    use std::sync::Arc;

    #[test]
    fn logical_bytes_uses_runtime_tensor_shapes() {
        let backend = ShapeOnlyBackend;
        let state = vec![
            interpreter::Value::Tensor(backend.zeros(catgrad::prelude::Shape(vec![2, 3]), catgrad::prelude::Dtype::F32)),
            interpreter::Value::Tensor(backend.zeros(catgrad::prelude::Shape(vec![5]), catgrad::prelude::Dtype::U32)),
        ];
        let snapshot = Snapshot::new(1, Arc::<str>::from("program"), state);
        assert_eq!(snapshot.logical_bytes(), (2 * 3 + 5) * 4);
    }
}
