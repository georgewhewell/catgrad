use catgrad::interpreter;
use catgrad_llm::types;
use catgrad_llm::utils::{get_model, get_model_chat_template, load_model};
use catgrad_llm::{BoundProgram, Detokenizer, PreparedPrompt, Program, Runtime};
use std::cell::RefCell;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub struct TextInferenceEngine<B: interpreter::Backend> {
    runtime: Runtime<B>,
    config_json: serde_json::Value,
    tokenizer: Tokenizer,
    chat_template: String,
    eos_token_ids: Vec<i32>,
    use_kv_cache: bool,
    bound_programs: RefCell<HashMap<usize, BoundProgram<B>>>,
}

#[allow(dead_code)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub termination: GenerationTermination,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTermination {
    Stop,
    MaxTokens,
}

impl From<GenerationTermination> for types::openai::FinishReason {
    fn from(value: GenerationTermination) -> Self {
        match value {
            GenerationTermination::Stop => Self::Stop,
            GenerationTermination::MaxTokens => Self::Length,
        }
    }
}

impl From<GenerationTermination> for types::anthropic::StopReason {
    fn from(value: GenerationTermination) -> Self {
        match value {
            GenerationTermination::Stop => Self::EndTurn,
            GenerationTermination::MaxTokens => Self::MaxTokens,
        }
    }
}

impl<B: interpreter::Backend> TextInferenceEngine<B> {
    pub fn new(
        model_name: &str,
        revision: &str,
        backend: B,
        use_kv_cache: bool,
    ) -> catgrad_llm::Result<Self> {
        let (parameter_values, parameter_types, config_json, tokenizer, _) =
            load_model(model_name, revision, &backend)?;
        let seed_program = Program::text_from_config(&config_json, 1)?;
        let runtime = Runtime::new(backend, &seed_program, parameter_values, parameter_types)?;
        let seed_bound = runtime.bind(seed_program)?;
        let chat_template = get_model_chat_template(model_name, revision)?
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");
        let eos_token_ids = get_model(&config_json, 1)?.config().get_eos_token_ids();

        let mut bound_programs = HashMap::new();
        bound_programs.insert(1, seed_bound);

        Ok(Self {
            runtime,
            config_json,
            tokenizer,
            chat_template,
            eos_token_ids,
            use_kv_cache,
            bound_programs: RefCell::new(bound_programs),
        })
    }

    pub fn prepare_messages(
        &self,
        messages: &[types::Message],
    ) -> catgrad_llm::Result<PreparedPrompt> {
        PreparedPrompt::from_messages(
            &self.tokenizer,
            &self.chat_template,
            messages,
            &self.eos_token_ids,
        )
    }

    #[allow(dead_code)]
    pub fn prepare_prompt(&self, prompt: &str) -> catgrad_llm::Result<PreparedPrompt> {
        PreparedPrompt::from_prompt(&self.tokenizer, prompt, &self.eos_token_ids)
    }

    pub fn generate_from_prepared<F>(
        &self,
        prepared: &PreparedPrompt,
        max_tokens: u32,
        mut on_text_delta: F,
    ) -> catgrad_llm::Result<GenerationOutput>
    where
        F: FnMut(&str) -> catgrad_llm::Result<()>,
    {
        let max_sequence_length = prepared.input_ids.len() + max_tokens as usize;
        let bound_program = self.bound_program(max_sequence_length)?;
        let empty_snapshot = bound_program.empty_snapshot();
        let mut snapshot = empty_snapshot.clone();
        let mut token_ids = encode_tokens(&prepared.input_ids)?;
        let mut decoder = Detokenizer::from_tokenizer(&self.tokenizer, &prepared.stop_token_ids);

        let mut completion_tokens = 0u32;
        let mut termination = GenerationTermination::MaxTokens;
        for _ in 0..max_tokens {
            let mut session = bound_program.start(snapshot)?;
            let token = session.step_text(&token_ids)?;
            let next_snapshot = session.into_snapshot();
            let decoded_token = i32::try_from(token).map_err(|_| {
                catgrad_llm::LLMError::UnsupportedWireConversion(format!(
                    "generated token id {token} exceeds i32 range"
                ))
            })?;

            let delta = decoder.push_tokens(&[decoded_token])?;
            if decoder.is_stopped() {
                termination = GenerationTermination::Stop;
                break;
            }

            completion_tokens += 1;
            if !delta.is_empty() {
                on_text_delta(&delta)?;
            }

            if self.use_kv_cache {
                snapshot = next_snapshot;
                token_ids = vec![token];
            } else {
                snapshot = empty_snapshot.clone();
                token_ids.push(token);
            }
        }

        Ok(GenerationOutput {
            text: decoder.finish(),
            prompt_tokens: prepared.input_ids.len() as u32,
            completion_tokens,
            termination,
        })
    }

    fn bound_program(&self, max_sequence_length: usize) -> catgrad_llm::Result<BoundProgram<B>> {
        if let Some(bound_program) = self.bound_programs.borrow().get(&max_sequence_length) {
            return Ok(bound_program.clone());
        }

        let program = Program::text_from_config(&self.config_json, max_sequence_length)?;
        let bound_program = self.runtime.bind(program)?;
        self.bound_programs
            .borrow_mut()
            .insert(max_sequence_length, bound_program.clone());
        Ok(bound_program)
    }
}

fn encode_tokens(token_ids: &[i32]) -> catgrad_llm::Result<Vec<u32>> {
    token_ids
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| {
                catgrad_llm::LLMError::UnsupportedWireConversion(format!(
                    "negative token id {token} cannot be encoded as u32"
                ))
            })
        })
        .collect()
}
