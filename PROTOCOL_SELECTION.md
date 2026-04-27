# Protocol Selection Design

Design note for `ToolParsingConfig`. Required before any second Qwen
dialect (Qwen3.5, qwen-xml, qwen-coder-xml) is added — the current
`tool_protocol_for(arch: &str)` registry is too coarse once multiple
parsers can serve the same architecture string.

This is **not** speculative code yet. It will be implemented when the
first incompatible second dialect is needed. Until then, the existing
single-row registry is correct (Qwen3 JSON only).

## Problem

vLLM ships at least three Qwen-family parsers:

- `qwen3` (their JSON-in-sentinel form — close to ours)
- `qwen3xml` / `qwen3coder` (XML-ish `<function=name><parameter=...>` form)
- `qwen35coder` (subclasses Qwen3-Coder, overrides streaming only)

These can all show up under architecture strings that overlap. Picking
a parser by `arch` alone risks binding the wrong one and silently
mis-parsing every tool call.

## Surface

```rust
// catgrad-llm::runtime::chat
pub struct ToolParsingConfig {
    pub protocol: ToolProtocolSelection,
    pub policy: ToolParsingPolicy,
    pub per_protocol: Option<ProtocolSpecificConfig>,
}

pub enum ToolProtocolSelection {
    Auto,                   // default
    Disabled,               // gateway-equivalent of "no tools"
    Named(ToolProtocolId),  // explicit override
}

pub enum ToolProtocolId {
    Qwen3Json,
    QwenXml,
    QwenCoderXml,
    Lfm2,
    Olmo3,
}

pub enum ToolParsingPolicy {
    Strict,      // current default; terminal errors on unknown / invalid
    // Permissive — DEFERRED, see below.
}

/// Per-protocol knobs. Each protocol module owns its own struct;
/// adding new knobs here doesn't affect other protocols' signatures.
pub enum ProtocolSpecificConfig {
    QwenXml(QwenXmlConfig),
    // ... future per-protocol variants
}

pub struct QwenXmlConfig {
    /// Some XML-emitting models start a call without the outer
    /// `<tool_call>` wrapper (just `<function=name>...`). vLLM's
    /// qwen3xml accepts this. We default to false (strict).
    pub allow_unwrapped_function: bool,
    // ... future XML knobs
}
```

`Default for ToolParsingConfig`: `Auto` + `Strict` + `None`. Today's
behavior is `Auto`-resolved via the existing `tool_protocol_for(arch)`
table (which still lives, just behind the resolver).

## Where it lives

Single resolution point: `ChatTurn::new`. Its signature gains one
optional parameter:

```rust
impl ChatTurn {
    pub fn new(
        arch: impl Into<String>,
        chat_template: Arc<str>,
        tokenizer: Arc<Tokenizer>,
        tokenizer_config: Arc<JsonValue>,
        stop_token_ids: Arc<[i32]>,
        tools: Option<Arc<ToolDirectory>>,
        options: ChatOptions,
        config: Option<ToolParsingConfig>,  // NEW; None = defaults
    ) -> Result<Self, ChatTurnConfigError>;
}
```

Resolution order (Auto):

1. Explicit override in `config.protocol = Named(...)` → honor.
2. Chat template inspection: a small list of marker strings keyed to
   `ToolProtocolId`. (E.g. template that emits literal `<tool_call>`
   with JSON body → `Qwen3Json`; template that emits `<function=name>`
   → `QwenXml`.) Cheap; scans the rendered template once.
3. Architecture fallback: today's `tool_protocol_for(arch)` lookup.
4. If `tools` is bound and we still have no protocol → `Err`.

`ChatTurnConfigError` grows one variant:
```rust
ToolsRequestedButProtocolAmbiguous {
    arch: String,
    candidates: Vec<ToolProtocolId>,
}
```
Returned when step 2 finds multiple template markers (e.g. both
`<tool_call>` and `<function=`) and the caller didn't disambiguate
via `Named(...)`.

## What the gateway sees

Nothing changes. Gateway code passes through whatever
`ToolParsingConfig` it's given (or `None`) and forwards to
`ModelAssets::chat_turn`. Gateway never touches `ToolProtocolId` or
parser internals — that's the point.

## Where the config comes from

For node: optionally per-deployment via a model-config block in
hellas-cli's config. Shape suggestion:

```toml
[model.tool_calling]
protocol = "auto"   # or "qwen3-json", "qwen-xml"
policy   = "strict"
# per_protocol can come later; YAGNI for now
```

`ModelAssets::chat_turn` reads this once at model load and passes the
resolved `Option<ToolParsingConfig>` into every `ChatTurn::new` call.

For serve.rs and llama: hardcoded defaults (`None` → Auto/Strict).
Examples don't need a config file.

## Permissive mode — deferred

`ToolParsingPolicy::Permissive` is the most attractive feature to add
speculatively and the most dangerous. The other agent's earlier
suggestion (`DecodeEvent::UnvalidatedToolCall`) is the right shape if
it ever lands, but defer until a concrete user appears. "Execute
hallucinated functions" creates a multi-year CVE tail.

## Implementation order when this work starts

1. Add `ToolParsingConfig` types (no behavior change yet — Auto resolves
   to today's table only).
2. Add the `config` parameter to `ChatTurn::new`. Existing callers pass
   `None`. All tests pass unchanged.
3. Implement chat-template inspection in the Auto resolver. Add unit
   tests for the resolver only.
4. Add the second Qwen parser as its own `ToolProtocolId` row. Wire its
   chat-template marker into the Auto resolver.
5. (If needed) Wire the `[model.tool_calling]` config into
   `hellas-cli`'s deployment config and through `ModelAssets`.

Step 4 is the only step that adds a parser. Steps 1-3 are infrastructure
that lands together as one PR; step 4 is a separate PR per dialect.

## Why not implement steps 1-3 now

They have no current user. The Auto resolver with one row is identical
to today's `tool_protocol_for(arch)`. Adding the type machinery without
a second protocol to disambiguate would be the kind of speculative
abstraction the FEEDBACK round just pruned.

When the second-Qwen-dialect work starts, this design lands as the
first PR of that effort. Until then, the table-with-one-row is correct.
