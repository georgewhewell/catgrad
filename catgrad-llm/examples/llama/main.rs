use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::*;
use catgrad_llm::{Program, ProgramInterface, Runtime};
use catgrad_llm::utils::{
    cache_path_for_embeddings, get_model, get_model_chat_template, load_and_preprocess_image,
    load_cached_embeddings, load_model, print_bench_table, render_chat_template,
    save_cached_embeddings,
};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(
        short = 'm',
        long,
        default_value = "HuggingFaceTB/SmolLM2-135M-Instruct"
    )]
    model_name: String,
    /// Model revision (branch, tag, or commit)
    #[arg(short = 'r', long, default_value = "main")]
    revision: String,
    /// TOML config file overriding model aliases.
    #[arg(short = 'c', long, value_name = "PATH")]
    config_file: Option<PathBuf>,
    /// List configured model aliases and exit
    #[arg(long)]
    list_models: bool,
    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,
    /// Optional image input for multimodal-capable models
    #[arg(short = 'i', long)]
    image: Option<PathBuf>,
    /// Pass raw prompt without chat template
    #[arg(long)]
    raw: bool,
    /// Tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    max_seq_len: usize,
    /// Use KV-cache
    #[arg(short = 'k', long)]
    kv_cache: bool,
    /// Enable typecheck
    #[arg(short = 't', long)]
    typecheck: bool,
    /// Backend to use
    #[arg(short = 'b', long, value_enum, default_value_t = BackendChoice::Candle)]
    backend: BackendChoice,
    /// Enable Candle backend acceleration
    #[arg(short = 'a', long)]
    accel: bool,
    /// Dump the constructed graph to this JSON file then exit.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Load model from a previously dumped JSON graph
    #[arg(long)]
    load: Option<PathBuf>,
    /// Benchmark
    #[arg(
        long,
        num_args = 2,
        value_names = ["PP", "TG"]
    )]
    bench: Option<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Candle,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    aliases: HashMap<String, String>,
}

fn parse_config(contents: &str, source: &str) -> Result<AppConfig> {
    let config: AppConfig = toml::from_str(contents)
        .map_err(|e| anyhow::anyhow!("invalid config file {source}: {e}"))?;
    Ok(config)
}

fn merge_config_file(app_config: &mut AppConfig, path: &Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read alias file {}: {e}", path.display()))?;
    let cfg = parse_config(&contents, &path.display().to_string())?;
    app_config.aliases.extend(cfg.aliases);
    Ok(())
}

// The app config currently contains only model aliases.
// The hardcoded ones can be overridden by user config files.
fn get_app_config(args: &Args) -> Result<AppConfig> {
    let default_config = include_str!("llm_config.default.toml");
    let mut app_config = parse_config(default_config, "embedded defaults")?;

    let local_alias_path = Path::new("llm_config.toml");
    if local_alias_path.exists() {
        merge_config_file(&mut app_config, local_alias_path)?;
    }

    if let Some(config_file) = &args.config_file {
        merge_config_file(&mut app_config, config_file)?;
    }
    Ok(app_config)
}

/// Construct, shapecheck, and interpret the a given LLM using the selected backend.
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let app_config = get_app_config(&args)?;
    if args.list_models {
        for (model, alias) in get_models(&app_config) {
            println!("{model} ({alias})");
        }
        return Ok(());
    }
    match args.backend {
        BackendChoice::Ndarray => run_with_backend(&args, &app_config, NdArrayBackend),
        BackendChoice::Candle => {
            run_with_backend(&args, &app_config, CandleBackend::new_accel(args.accel))
        }
    }
}

fn get_model_name(args: &Args, app_config: &AppConfig) -> Result<String> {
    Ok(app_config
        .aliases
        .get(args.model_name.as_str())
        .cloned()
        .unwrap_or_else(|| args.model_name.clone()))
}

fn get_models(app_config: &AppConfig) -> Vec<(&str, &str)> {
    let mut models: Vec<(&str, &str)> = app_config
        .aliases
        .iter()
        .map(|(alias, model)| (model.as_str(), alias.as_str()))
        .collect();
    models.sort_unstable_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    models
}

fn run_with_backend<B: interpreter::Backend>(
    args: &Args,
    app_config: &AppConfig,
    backend: B,
) -> Result<()> {
    let model_name = get_model_name(args, app_config)?;

    let start_load = std::time::Instant::now();
    let (parameter_values, parameter_types, config_json, tokenizer, total_params) =
        load_model(&model_name, &args.revision, &backend)?;
    let elapsed_load = start_load.elapsed();

    eprintln!(
        "Model weights loaded for {} in {:.2} seconds",
        model_name,
        elapsed_load.as_secs_f64()
    );

    let chat_template = get_model_chat_template(&model_name, &args.revision).unwrap_or_default();

    let benchmarking = args.bench.is_some();
    let mut pp = 0;
    let mut tg = 0;
    let mut max_seq_len = args.max_seq_len;
    let use_image = args.image.is_some() && !benchmarking;

    let prompt = if let Some(bench) = &args.bench {
        pp = bench[0];
        tg = bench[1];
        max_seq_len = tg;
        eprintln!(
            "Benchmarking {} with prefill size {} and sequence length {}",
            &model_name, pp, tg
        );
        "The".repeat(pp)
    } else if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(&chat_template, &args.prompt, use_image, false)?
    };

    if !benchmarking {
        print!("{}", prompt);
    }
    let prompt = if use_image {
        // FIXME: this extra get_model call is needed to get the multimodal prompt interpolation length.
        // maybe allow max_sequence_length be set after the constructor
        let model = get_model(&config_json, 1)?;
        if !model.is_multimodal() {
            return Err(anyhow::anyhow!(
                "Model {} does not support image input",
                model_name
            ));
        }
        model
            .multimodal_interpolate_prompt(&prompt)
            .ok_or_else(|| {
                anyhow::anyhow!("Model did not provide multimodal prompt interpolation")
            })?
    } else {
        prompt
    };

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let max_sequence_length = max_seq_len + token_ids.len();
    let model = get_model(&config_json, max_sequence_length)?;

    let mm_metadata = if use_image {
        Some(
            model
                .multimodal_metadata()
                .ok_or_else(|| anyhow::anyhow!("Model {} is not multimodal", model_name))?,
        )
    } else {
        None
    };

    let program = if let Some(load_path) = &args.load {
        let file = std::fs::File::open(load_path)?;
        serde_json::from_reader(file)?
    } else if use_image {
        let language_model = model.multimodal_language_module().ok_or_else(|| {
            anyhow::anyhow!(
                "Model {} does not provide multimodal language module",
                model_name
            )
        })?;
        Program::from_module(
            language_model.as_ref(),
            ProgramInterface::Raw,
            catgrad::prelude::Path::empty(),
            model.empty_state_type(),
            max_sequence_length,
            model.weight_post_process(),
        )?
    } else {
        Program::text_from_config(&config_json, max_sequence_length)?
    };

    if let Some(dump_path) = &args.dump {
        let file = std::fs::File::create(dump_path)?;
        serde_json::to_writer_pretty(file, &program)?;
        eprintln!(
            "Program for {} and max_seq_length of {max_sequence_length} dumped to {}",
            model.path(),
            dump_path.display()
        );
        return Ok(());
    }

    let mut generated_tokens = 0;
    let mut start_gen = std::time::Instant::now();
    let mut elapsed_pp = std::time::Duration::ZERO;
    let runtime = Runtime::new(backend, &program, parameter_values, parameter_types)?;
    let bound_program = runtime.bind(program)?;

    let mut multimodal_ctx: Option<MultimodalRuntime<B>> = None;
    if let Some(mm) = mm_metadata {
        let vision_model = model.multimodal_vision_module().ok_or_else(|| {
            anyhow::anyhow!("Model {} does not provide vision module", model_name)
        })?;
        let vision_program = Program::from_module(
            vision_model.as_ref(),
            ProgramInterface::Raw,
            catgrad::prelude::Path::empty(),
            vec![],
            0,
            model.weight_post_process(),
        )?;
        let bound_vision = runtime.bind(vision_program)?;
        let image_path = args
            .image
            .as_ref()
            .expect("image existence already checked");
        let (image_data, image_shape) =
            load_and_preprocess_image(image_path, mm.image_size, mm.patch_size)?;
        let cache_path =
            cache_path_for_embeddings(&model_name, &image_path.to_string_lossy(), &image_data);
        let visual_embeddings = if let Ok(cached) = load_cached_embeddings(&cache_path) {
            eprintln!(
                "Loading cached image features from: {}",
                cache_path.display()
            );
            interpreter::tensor(
                runtime.backend(),
                Shape(vec![1, mm.mm_tokens_per_image, mm.hidden_size]),
                cached,
            )
            .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?
        } else {
            let image_tensor = interpreter::tensor(runtime.backend(), Shape(image_shape), image_data)
                .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;
            let mut vision_session = bound_vision.start(bound_vision.empty_snapshot())?;
            let mut results = vision_session.run_raw(vec![image_tensor])?;
            if results.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Vision program returned {} non-state outputs",
                    results.len()
                ));
            }
            let visual_embeddings = results.remove(0);
            let flattened = to_f32_vec(runtime.backend(), &visual_embeddings)?;
            save_cached_embeddings(&cache_path, &flattened)?;
            eprintln!("Saved image features to: {}", cache_path.display());
            visual_embeddings
        };

        multimodal_ctx = Some(MultimodalRuntime {
            hidden_size: mm.hidden_size,
            image_token_index: mm.image_token_index,
            visual_embeddings,
        });
    }

    let eos_token_ids = model.config().get_eos_token_ids();

    let use_kv_cache = args.kv_cache || use_image;
    let empty_snapshot = bound_program.empty_snapshot();
    let mut snapshot = empty_snapshot.clone();
    let mut use_image_embeddings = use_image;

    // Run inference loop
    for i in 0..max_seq_len {
        let mut session = bound_program.start(snapshot)?;
        let next_token_id = if let Some(ctx) = multimodal_ctx.as_ref() {
            let outputs = session.run_raw(build_multimodal_inputs(
                runtime.backend(),
                &token_ids,
                ctx.hidden_size,
                ctx.image_token_index,
                &ctx.visual_embeddings,
                use_image_embeddings,
            )?)?;
            let mut outputs = outputs;
            if outputs.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Language program returned {} non-state outputs",
                    outputs.len()
                ));
            }
            extract_generated_token(runtime.backend(), outputs.remove(0))?
        } else {
            session.step_text(&token_ids)?
        };
        let next_snapshot = session.into_snapshot();
        if i == 0 {
            elapsed_pp = start_gen.elapsed();
            start_gen = std::time::Instant::now();
        }
        generated_tokens += 1;
        if eos_token_ids.contains(&(next_token_id as i32)) && !benchmarking {
            break;
        }
        if use_kv_cache {
            snapshot = next_snapshot;
            token_ids = vec![next_token_id];
        } else {
            snapshot = empty_snapshot.clone();
            token_ids.push(next_token_id);
        }
        if use_image && use_kv_cache {
            use_image_embeddings = false;
        }
        if !benchmarking {
            let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
            print!("{}", decoded_token);
            std::io::stdout().flush()?;
        }
    }

    let elapsed_gen = start_gen.elapsed();
    if benchmarking {
        // hardcode size multiplier as 4.0 since we only load in F32
        let size_gib = (total_params as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        let params_m = total_params as f64 / 1_000_000.0;
        let b_str = match args.backend {
            BackendChoice::Ndarray => "Ndarray",
            BackendChoice::Candle => "Candle",
        };
        print_bench_table(
            &model_name,
            size_gib,
            params_m,
            b_str,
            pp,
            elapsed_pp,
            tg,
            elapsed_gen,
        );
    } else {
        println!();
        eprintln!(
            "{} tokens generated in {} seconds. ({:.2} tps)",
            generated_tokens,
            (elapsed_pp + elapsed_gen).as_secs(),
            generated_tokens as f64 / (elapsed_pp + elapsed_gen).as_secs_f64(),
        );
    }
    Ok(())
}

fn to_f32_vec<B: interpreter::Backend>(
    backend: &B,
    value: &interpreter::Value<B>,
) -> Result<Vec<f32>> {
    match value.clone() {
        interpreter::Value::Tensor(arr) => match backend.to_vec(arr) {
            interpreter::TaggedVec::F32(v) => Ok(v),
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}

struct MultimodalRuntime<B: interpreter::Backend> {
    hidden_size: usize,
    image_token_index: usize,
    visual_embeddings: interpreter::Value<B>,
}

fn extract_generated_token<B: interpreter::Backend>(
    backend: &B,
    output: interpreter::Value<B>,
) -> Result<u32> {
    match output {
        interpreter::Value::Tensor(arr) => match backend.to_vec(arr) {
            interpreter::TaggedVec::U32(v) => {
                let token = v
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("token output tensor was empty"))?;
                Ok(token)
            }
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}

fn build_multimodal_inputs<B: interpreter::Backend>(
    backend: &B,
    input_tokens: &[u32],
    hidden_size: usize,
    image_token_index: usize,
    visual_embeddings: &interpreter::Value<B>,
    use_image_embeddings: bool,
) -> Result<Vec<interpreter::Value<B>>> {
    let empty_image_embeddings = interpreter::tensor(
        backend,
        Shape(vec![1, 0, hidden_size]),
        Vec::<f32>::new(),
    )
    .map_err(|err| anyhow::anyhow!("empty image tensor error: {:?}", err))?;

    let (text_before_tokens, text_after_tokens) = if use_image_embeddings {
        let first_image_token_index = input_tokens
            .iter()
            .position(|&x| x == image_token_index as u32)
            .unwrap_or(0);
        let last_image_token_index = input_tokens
            .iter()
            .rposition(|&x| x == image_token_index as u32)
            .unwrap_or(0);
        (
            &input_tokens[..first_image_token_index],
            &input_tokens[last_image_token_index + 1..],
        )
    } else {
        (&[][..], input_tokens)
    };

    let text_before = interpreter::tensor(
        backend,
        Shape(vec![1, text_before_tokens.len()]),
        text_before_tokens.to_vec(),
    )
    .map_err(|err| anyhow::anyhow!("text_before tensor error: {:?}", err))?;
    let text_after = interpreter::tensor(
        backend,
        Shape(vec![1, text_after_tokens.len()]),
        text_after_tokens.to_vec(),
    )
    .map_err(|err| anyhow::anyhow!("text_after tensor error: {:?}", err))?;
    let image_embeddings = if use_image_embeddings {
        visual_embeddings.clone()
    } else {
        empty_image_embeddings
    };

    Ok(vec![text_before, image_embeddings, text_after])
}
