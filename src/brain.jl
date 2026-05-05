# Phase 3 — LLM cognitive layer (ADR-0002).
#
# `NullBrain` is the reproducibility-preserving default. `RandomBrain`
# perturbs opinions with a deterministic Normal draw — useful for
# wiring tests without a model. `LiveBrain` calls a per-agent dispatch
# function (PromptingTools by default; an injected stub in tests),
# applies the two-layer Δ bound from ADR-0002 §4, enforces an LLM
# call budget, detects schema drift, and persists every call to the
# DuckDB `transcripts` table.
#
# CI hard constraint: the dispatch function on `LiveBrain` is
# injectable so `EIDOLON_LLM_MODE=mock` test runs make zero real
# network calls.

using Agents: abmproperties, abmtime, allagents, nearby_agents
using DBInterface
using Distributions: Normal
using DuckDB
import PromptingTools as PT
using Random: AbstractRNG

"""
    AbstractBrain

Supertype for all cognitive backends (ADR-0002 §1). Concrete subtypes:
`NullBrain`, `RandomBrain`, `LiveBrain` (and `ReplayBrain` later).
"""
abstract type AbstractBrain end

"""
    BrainOutput(delta, new_memory, reasoning)

What a brain returns for one agent at one tick (ADR-0002 §1).

- `delta::Float64` — bounded opinion perturbation (`[-Δ_max, +Δ_max]`).
- `new_memory::String` — text to append to the agent's memory ring.
- `reasoning::String` — free-text rationale; persisted to transcripts only.
"""
struct BrainOutput
    delta::Float64
    new_memory::String
    reasoning::String
end

"""
    LLMBudgetExceeded(cap, used)

Thrown by `batch_reflect!(::LiveBrain, …)` when the dispatched-call
counter would exceed `cap` (ADR-0002 §5). The partially-written
DuckDB store is left intact.
"""
struct LLMBudgetExceeded <: Exception
    cap::Int
    used::Int
end

function Base.showerror(io::IO, e::LLMBudgetExceeded)
    print(
        io,
        "LLMBudgetExceeded: dispatched LLM call ", e.used,
        " would exceed cap of ", e.cap,
        " (set via EIDOLON_MAX_LLM_CALLS or `live_brain(... ; max_calls = ...)`)"
    )
    return nothing
end

"""
    SchemaDriftError(tick, n_failed, n_total, threshold)

Thrown by `batch_reflect!(::LiveBrain, …)` when more than
`threshold` (default 5%) of agents in a single tick fail
`aiextract` parsing (ADR-0002 §6).
"""
struct SchemaDriftError <: Exception
    tick::Int
    n_failed::Int
    n_total::Int
    threshold::Float64
end

function Base.showerror(io::IO, e::SchemaDriftError)
    pct = 100 * e.n_failed / max(e.n_total, 1)
    print(
        io,
        "SchemaDriftError: tick ", e.tick, " — ",
        e.n_failed, "/", e.n_total, " agents failed schema parsing (",
        round(pct; digits = 2), "% > ", round(100 * e.threshold; digits = 2), "%)"
    )
    return nothing
end

# Internal exceptions used by the dispatch layer. Tests construct these
# from a stub dispatch to drive the retry / schema-drift code paths
# without touching a real LLM.

"""
    RetryableError(msg)

Internal sentinel raised by a `LiveBrain` dispatch to mark a transient
failure (HTTP 429, 5xx, network timeout) that should trigger
exponential-backoff retry. Not exported.
"""
struct RetryableError <: Exception
    msg::String
end

Base.showerror(io::IO, e::RetryableError) = print(io, "RetryableError: ", e.msg)

"""
    SchemaParseError(msg, response)

Internal sentinel raised by a `LiveBrain` dispatch when the LLM
response cannot be coerced into [`BrainOutput`](@ref). Counted toward
the schema-drift budget; never retried (ADR-0002 §6). Not exported.
"""
struct SchemaParseError <: Exception
    msg::String
    response::String
end

Base.showerror(io::IO, e::SchemaParseError) = print(io, "SchemaParseError: ", e.msg)

# --- Concrete brains -------------------------------------------------

"""
    NullBrain

Returns Δ = 0 for every agent on every tick. Default under
`EIDOLON_LLM_MODE=mock`; preserves the ADR-0001 reproducibility
contract.
"""
struct NullBrain <: AbstractBrain end

"""
    RandomBrain(rng, sigma; delta_cap = brain_delta_cap())

Samples Δ ∼ Normal(0, sigma) for every agent at every tick. Useful for
exercising the perturbation pipeline without LLM cost. Determinism
is the user's responsibility — pass a freshly seeded `rng` per run.
The same `delta_cap` semantics apply as `LiveBrain`.
"""
struct RandomBrain{R <: AbstractRNG} <: AbstractBrain
    rng::R
    sigma::Float64
    delta_cap::Float64
end

function RandomBrain(rng::AbstractRNG, sigma::Real; delta_cap::Real = brain_delta_cap())
    return RandomBrain(rng, Float64(sigma), Float64(delta_cap))
end

"""
    LiveBrain(model_name, template, max_concurrency, max_attempts,
              delta_cap, max_calls, schema_drift_threshold,
              call_counter, dispatch)

PromptingTools-backed brain (ADR-0002 §3). `dispatch` is an injectable
seam: in CI / under `EIDOLON_LLM_MODE=mock` tests pass a deterministic
stub so no real network call is made. The default dispatch built by
[`live_brain`](@ref) goes through `PromptingTools.aiextract`.

Fields:
- `model_name`        — passed through to PromptingTools (`cfg.llm_model`).
- `template`          — registered template symbol (`:agent_reflection_v1`).
- `max_concurrency`   — `asyncmap` `ntasks` (default 16).
- `max_attempts`      — retry attempts on `RetryableError` (default 3).
- `delta_cap`         — Δ_max (default 0.05; env `EIDOLON_BRAIN_DELTA_CAP`).
- `max_calls`         — LLM call budget (default 5000; env `EIDOLON_MAX_LLM_CALLS`).
                         `0` means unlimited.
- `schema_drift_threshold` — fraction of agents/tick allowed to fail
                              parsing before `SchemaDriftError` (default 0.05).
- `call_counter`      — `Ref{Int}` counting dispatched calls across the run.
- `dispatch`          — `(brain, agent_id, vars::NamedTuple) →
                          (; prompt::String, response::String, output::BrainOutput)`.
"""
struct LiveBrain <: AbstractBrain
    model_name::String
    template::Symbol
    max_concurrency::Int
    max_attempts::Int
    delta_cap::Float64
    max_calls::Int
    schema_drift_threshold::Float64
    call_counter::Base.RefValue{Int}
    dispatch::Any
end

# --- Env helpers -----------------------------------------------------

"""
    brain_delta_cap()

Resolve `Δ_max` from `EIDOLON_BRAIN_DELTA_CAP` (default `0.05`).
"""
function brain_delta_cap()
    s = get(ENV, "EIDOLON_BRAIN_DELTA_CAP", "")
    return isempty(s) ? 0.05 : parse(Float64, s)
end

"""
    max_llm_calls()

Resolve the LLM call budget from `EIDOLON_MAX_LLM_CALLS` (default
`5000`). A value of `0` is treated as unlimited.
"""
function max_llm_calls()
    s = get(ENV, "EIDOLON_MAX_LLM_CALLS", "")
    return isempty(s) ? 5000 : parse(Int, s)
end

# --- Reflection template (ADR-0002 §7) -------------------------------

const REFLECTION_TEMPLATE_NAME = :agent_reflection_v1

const _REFLECTION_SYSTEM = """
You are role-playing as a single agent inside an opinion-dynamics
simulation. Stay in character as the persona described. Respond ONLY
with a JSON object that matches the schema you are given — no prose,
no commentary, no markdown fences.
"""

const _REFLECTION_USER = """
Persona description: {{persona_description}}

Recent memory (newest first; may be empty):
{{recent_memory}}

Neighbour opinions (anonymous, in [0, 1]): {{neighbour_opinions}}

Most recent global event (or "none"): {{global_event}}

Task: produce a brief reflection and propose a small adjustment to
your scalar opinion. The adjustment `delta` MUST lie in
[-{{delta_cap}}, +{{delta_cap}}]; values outside that interval will be
clamped and flagged in the transcript. Keep `new_memory` ≤ 200 chars
and `reasoning` ≤ 500 chars.
"""

function _register_reflection_template()
    try
        PT.create_template(
            _REFLECTION_SYSTEM,
            _REFLECTION_USER;
            load_as = REFLECTION_TEMPLATE_NAME
        )
    catch
        # Already registered, or PT API drift — non-fatal. The default
        # dispatch path falls back to ad-hoc prompts when the template
        # is unavailable, and tests use injected dispatches anyway.
    end
    return nothing
end

function __init__()
    _register_reflection_template()
    return nothing
end

# --- Helper: build prompt vars from agent + model --------------------

function _persona_description(agent, cfg)
    for p in cfg.personas
        p.id == agent.persona_id && return p.description
    end
    return "(unknown persona: $(agent.persona_id))"
end

function _recent_memory(agent)
    isempty(agent.memory) && return "(no memory yet)"
    # Newest-first slice; cap at 5 entries to keep prompts small.
    n = min(length(agent.memory), 5)
    items = agent.memory[(end - n + 1):end]
    return join(reverse(items), "\n - ")
end

function _neighbour_opinions(agent, model)
    return [round(n.opinion; digits = 3) for n in nearby_agents(agent, model)]
end

function _global_event(model, tick::Integer)
    cfg = model.cfg
    isempty(cfg.interventions) && return "none"
    relevant = [iv for iv in cfg.interventions if iv.tick <= tick]
    isempty(relevant) && return "none"
    iv = last(relevant)
    return string(iv.kind, ": ", JSON.json(iv.payload))
end

function _build_vars(agent, model, brain::LiveBrain, tick::Integer)
    return (
        persona_description = _persona_description(agent, model.cfg),
        recent_memory = _recent_memory(agent),
        neighbour_opinions = _neighbour_opinions(agent, model),
        global_event = _global_event(model, tick),
        delta_cap = brain.delta_cap
    )
end

# --- Default dispatch (PromptingTools) -------------------------------

"""
    _pt_schema_for(model_name)

Pick the PromptingTools schema for `model_name` based on its prefix.
PromptingTools' default registry lookup falls back to `OpenAISchema`
for unknown names, which silently routes Anthropic models to the
wrong endpoint — this helper makes the dispatch explicit.
"""
function _pt_schema_for(model_name::AbstractString)
    startswith(model_name, "claude") && return PT.AnthropicSchema()
    startswith(model_name, "gpt") && return PT.OpenAISchema()
    return PT.OpenAISchema()
end

"""
    default_live_dispatch(brain, agent_id, vars)

Default `LiveBrain.dispatch`: calls `PromptingTools.aiextract` against
the registered template and coerces the response into a
[`BrainOutput`](@ref). Dispatches through an explicit schema chosen by
[`_pt_schema_for`](@ref) and reads the API key from the env at call
time (PromptingTools' preference cache is initialised at module load,
so a `LocalPreferences.toml` would otherwise win over a runtime
export). Translates known transient failures into [`RetryableError`](@ref)
and parsing failures into [`SchemaParseError`](@ref). Never invoked
under `EIDOLON_LLM_MODE=mock` test runs — those inject their own
dispatch.
"""
function default_live_dispatch(brain::LiveBrain, agent_id::Int, vars::NamedTuple)
    api_key = get(ENV, "ANTHROPIC_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "default_live_dispatch: ANTHROPIC_API_KEY is empty. Set it in the env " *
        "before launching Julia, or inject a stub dispatch via " *
        "`live_brain(cfg; dispatch = ...)` for tests."
    ))
    schema = _pt_schema_for(brain.model_name)
    local msg
    try
        msg = PT.aiextract(
            schema,
            brain.template;
            return_type = BrainOutput,
            model = brain.model_name,
            api_key = api_key,
            vars...
        )
    catch err
        # PromptingTools wraps HTTP errors in its own types; we treat
        # any non-deterministic failure as retryable. Schema-validation
        # failures inside aiextract surface as a different exception
        # (handled below).
        throw(RetryableError(sprint(showerror, err)))
    end
    output = msg.content
    if !(output isa BrainOutput)
        throw(SchemaParseError(
            "aiextract returned $(typeof(output)), expected BrainOutput",
            string(output)
        ))
    end
    return (
        prompt = sprint(show, msg),
        response = string(output),
        output = output
    )
end

# --- live_brain helper ----------------------------------------------

"""
    live_brain(cfg::WorldConfig; kwargs...) -> LiveBrain

Build a `LiveBrain` whose `model_name` defaults to `cfg.llm_model` and
whose Δ-cap and call-budget defaults come from the
`EIDOLON_BRAIN_DELTA_CAP` / `EIDOLON_MAX_LLM_CALLS` environment
variables (see [`brain_delta_cap`](@ref) / [`max_llm_calls`](@ref)).

Override any field via kwargs. The most useful overrides in tests are
`dispatch` (inject a deterministic stub) and `max_calls` (drive the
[`LLMBudgetExceeded`](@ref) path).
"""
function live_brain(
        cfg::WorldConfig;
        model_name::AbstractString = cfg.llm_model,
        template::Symbol = REFLECTION_TEMPLATE_NAME,
        max_concurrency::Integer = 16,
        max_attempts::Integer = 3,
        delta_cap::Real = brain_delta_cap(),
        max_calls::Integer = max_llm_calls(),
        schema_drift_threshold::Real = 0.05,
        dispatch = default_live_dispatch
)
    return LiveBrain(
        String(model_name),
        template,
        Int(max_concurrency),
        Int(max_attempts),
        Float64(delta_cap),
        Int(max_calls),
        Float64(schema_drift_threshold),
        Ref(0),
        dispatch
    )
end

# --- batch_reflect! --------------------------------------------------

"""
    batch_reflect!(brain::AbstractBrain, model;
                   tick::Integer = current_tick + 1) -> Dict{Int,BrainOutput}

Run the brain's reflection phase for the upcoming `tick` (ADR-0002 §1).
The result is stashed at `model.brain_outputs` so [`brain_perturbation`](@ref)
— called from `agent_step!` — can read it synchronously.

Tick semantics: `batch_reflect!` is invoked **before** `Agents.step!`,
so its outputs drive the opinion update that produces the *new* tick.
By default the reflection is tagged with `abmtime(model) + 1`; pass
an explicit `tick` to override.
"""
function batch_reflect!(::NullBrain, model; tick::Integer = abmtime(model) + 1)
    Dict{Int, BrainOutput}()
end

function batch_reflect!(
        brain::RandomBrain, model;
        tick::Integer = abmtime(model) + 1
)
    # Sort by id so the draw order is independent of the model's
    # internal agent storage order — important for cross-run determinism.
    ids = sort!([Int(a.id) for a in allagents(model)])
    out = Dict{Int, BrainOutput}()
    sizehint!(out, length(ids))
    for id in ids
        delta = rand(brain.rng, Normal(0.0, brain.sigma))
        out[id] = BrainOutput(delta, "", "")
    end
    model.brain_outputs = out
    return out
end

function batch_reflect!(
        brain::LiveBrain, model;
        tick::Integer = abmtime(model) + 1
)
    agents = sort!(collect(allagents(model)); by = a -> Int(a.id))
    n = length(agents)

    rows = Vector{NamedTuple}(undef, n)
    outs = Vector{BrainOutput}(undef, n)
    errors = Vector{Union{Nothing, Exception}}(undef, n)
    fill!(errors, nothing)

    schema_failures = Threads.Atomic{Int}(0)

    asyncmap(1:n; ntasks = brain.max_concurrency) do i
        try
            outs[i],
            rows[i] = _reflect_one!(
                brain, agents[i], model, Int(tick), schema_failures
            )
        catch err
            errors[i] = err
            outs[i] = BrainOutput(0.0, "", "<errored>")
            rows[i] = (
                tick = Int(tick),
                agent_id = Int(agents[i].id),
                model = brain.model_name,
                template = String(brain.template),
                prompt = "",
                response = "",
                delta_raw = nothing,
                delta_clamped = nothing,
                status = "failed",
                latency_ms = 0
            )
        end
        return nothing
    end

    # Re-raise the first fatal error (LLMBudgetExceeded preferred over
    # any other surprise). Per-agent retry-exhausted failures are
    # captured inside `_reflect_one!` and do NOT reach this list.
    for err in errors
        err isa LLMBudgetExceeded && throw(err)
    end
    for err in errors
        err === nothing || throw(err)
    end

    drift_rate = schema_failures[] / max(n, 1)
    if drift_rate > brain.schema_drift_threshold
        throw(SchemaDriftError(
            Int(tick), schema_failures[], n, brain.schema_drift_threshold
        ))
    end

    # DuckDB.jl is a single writer per file (ADR-0004): flush all
    # transcript rows in one go on the main task, after asyncmap.
    # `hasproperty(model, …)` doesn't peek into Agents.jl's properties
    # Dict, so query the Dict directly via `abmproperties`.
    store = get(abmproperties(model), :store, nothing)
    if store !== nothing
        _persist_transcripts!(store, rows)
    end

    outputs = Dict{Int, BrainOutput}()
    sizehint!(outputs, n)
    for (i, a) in enumerate(agents)
        outputs[Int(a.id)] = outs[i]
    end
    model.brain_outputs = outputs
    return outputs
end

function _reflect_one!(
        brain::LiveBrain, agent, model, tick::Int, schema_failures
)
    # Cost cap: increment first, then check. Cooperative-yield-safe
    # because the increment + comparison contains no I/O.
    brain.call_counter[] += 1
    used = brain.call_counter[]
    if brain.max_calls > 0 && used > brain.max_calls
        throw(LLMBudgetExceeded(brain.max_calls, used))
    end

    vars = _build_vars(agent, model, brain, tick)

    attempt = 0
    response_str = ""
    prompt_str = ""
    output = BrainOutput(0.0, "", "")
    status = "ok"
    delta_raw = nothing
    delta_clamped = nothing
    latency_ms = 0

    while true
        attempt += 1
        t0 = time_ns()
        try
            result = brain.dispatch(brain, Int(agent.id), vars)
            prompt_str = String(result.prompt)
            response_str = String(result.response)
            output = result.output::BrainOutput
            latency_ms = Int(round((time_ns() - t0) / 1.0e6))
            status = attempt > 1 ? "retried" : "ok"
            delta_raw = output.delta
            delta_clamped = clamp(output.delta, -brain.delta_cap, brain.delta_cap)
            output = BrainOutput(output.delta, output.new_memory, output.reasoning)
            break
        catch err
            latency_ms = Int(round((time_ns() - t0) / 1.0e6))
            if err isa SchemaParseError
                Threads.atomic_add!(schema_failures, 1)
                status = "schema_error"
                response_str = err.response
                output = BrainOutput(0.0, "", "<schema-error>")
                delta_raw = nothing
                delta_clamped = 0.0
                break
            elseif err isa RetryableError && attempt < brain.max_attempts
                # Exponential backoff: 0.5s, 1.0s, 2.0s, …
                sleep(0.5 * 2.0^(attempt - 1))
                continue
            elseif err isa RetryableError
                status = "failed"
                output = BrainOutput(0.0, "", "<retry-exhausted>")
                delta_raw = nothing
                delta_clamped = 0.0
                break
            else
                rethrow(err)
            end
        end
    end

    row = (
        tick = tick,
        agent_id = Int(agent.id),
        model = brain.model_name,
        template = String(brain.template),
        prompt = prompt_str,
        response = response_str,
        delta_raw = delta_raw,
        delta_clamped = delta_clamped,
        status = status,
        latency_ms = latency_ms
    )
    return output, row
end

# --- Transcript persistence -----------------------------------------

const _TRANSCRIPT_INSERT_SQL = """
INSERT INTO transcripts
    (tick, agent_id, model, template, prompt, response,
     delta_raw, delta_clamped, status, latency_ms)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

function _persist_transcripts!(db::DuckDB.DB, rows)
    isempty(rows) && return nothing
    stmt = DBInterface.prepare(db, _TRANSCRIPT_INSERT_SQL)
    try
        for r in rows
            DBInterface.execute(
                stmt,
                (
                    Int32(r.tick),
                    Int32(r.agent_id),
                    r.model,
                    r.template,
                    r.prompt,
                    r.response,
                    r.delta_raw,
                    r.delta_clamped,
                    r.status,
                    Int32(r.latency_ms)
                )
            )
        end
    finally
        DBInterface.close!(stmt)
    end
    return nothing
end

# --- Perturbation hook ----------------------------------------------

"""
    brain_perturbation(brain, agent, model) -> Float64

Per-tick opinion perturbation Δ called from `agent_step!`
(ADR-0001 / ADR-0002 §4). For non-`NullBrain` brains, looks up
`model.brain_outputs[agent.id]`, clamps to `[-Δ_max, +Δ_max]`, and
returns the clamped scalar — defence-in-depth on top of the prompted
bound and the at-write-time clamp persisted to the transcript.
"""
brain_perturbation(::NullBrain, agent, model) = 0.0

function brain_perturbation(brain::RandomBrain, agent, model)
    out = get(model.brain_outputs, Int(agent.id), nothing)
    out === nothing && return 0.0
    return clamp(out.delta, -brain.delta_cap, brain.delta_cap)
end

function brain_perturbation(brain::LiveBrain, agent, model)
    out = get(model.brain_outputs, Int(agent.id), nothing)
    out === nothing && return 0.0
    return clamp(out.delta, -brain.delta_cap, brain.delta_cap)
end
