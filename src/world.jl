# Phase 2 — GraphSpace, world initialisation, and the headless run loop.

using Agents
using DataFrames
using Distributions: Categorical, Normal
using Graphs: watts_strogatz
using Random: Xoshiro

"""
    initialize_world(cfg::WorldConfig; brain::AbstractBrain = NullBrain())
        -> StandardABM

Build a `StandardABM` from a [`WorldConfig`](@ref):

- Watts–Strogatz small-world graph (Graphs.jl) wrapped in a `GraphSpace`.
- One `EidolonAgent` per graph vertex; its persona is sampled from
  `cfg.persona_distribution`, its opinion from
  `Normal(persona.opinion_prior_mean, persona.opinion_prior_std)`
  clamped to `[0, 1]`.
- Model RNG is `Xoshiro(cfg.seed)` — the only stochasticity source
  under `NullBrain` (ADR-0001 reproducibility contract).
- `cfg` and `brain` live in the model's properties as `model.cfg` /
  `model.brain` so `agent_step!` can reach them.
"""
function initialize_world(cfg::WorldConfig; brain::AbstractBrain = NullBrain())
    cfg.topology.kind == "watts_strogatz" || throw(ArgumentError(
        "initialize_world: unsupported topology kind \"$(cfg.topology.kind)\" " *
        "(v1 supports: watts_strogatz)",
    ))

    rng = Xoshiro(cfg.seed)

    k = Int(cfg.topology.params["k"])
    β = Float64(cfg.topology.params["beta"])
    graph = watts_strogatz(cfg.n_agents, k, β; rng = rng)
    space = GraphSpace(graph)

    persona_by_id = Dict(p.id => p for p in cfg.personas)
    properties = Dict{Symbol, Any}(
        :cfg => cfg,
        :brain => brain,
        :brain_outputs => Dict{Int, BrainOutput}(),
        :store => nothing
    )

    model = StandardABM(
        EidolonAgent,
        space;
        agent_step! = agent_step!,
        rng = rng,
        properties = properties
    )

    # Sort persona ids for deterministic sampling — Dict iteration order
    # is not part of the language guarantee we want to lean on.
    persona_ids = sort!(collect(keys(cfg.persona_distribution)))
    weights = [cfg.persona_distribution[id] for id in persona_ids]
    sampler = Categorical(weights)

    for v in 1:cfg.n_agents
        idx = rand(rng, sampler)
        pid = persona_ids[idx]
        persona = persona_by_id[pid]
        raw = rand(rng, Normal(persona.opinion_prior_mean, persona.opinion_prior_std))
        opinion = clamp(raw, 0.0, 1.0)
        add_agent!(
            v,
            model,
            opinion,
            persona.confidence_radius,
            persona.update_weight,
            pid,
            String[]
        )
    end

    return model
end

"""
    run_simulation(cfg::WorldConfig;
        brain::AbstractBrain = NullBrain(),
        n_steps::Integer = cfg.max_ticks,
        run_id::Union{Nothing, AbstractString} = nothing,
    )::DataFrame

Build a model from `cfg`, run it for `n_steps`, and return a
`DataFrame` with one row per agent per tick. Columns:
`tick`, `agent_id`, `opinion`, `persona_id`. Tick 0 is the initial
state; tick `n_steps` is the post-final-step state.

When `run_id` is given, the run is also persisted to
`runs/<run_id>/store.duckdb` per ADR-0004: `meta` and `personas` are
written once at start, then [`flush_tick!`](@ref) is called after each
step (including a tick-0 dump of the initial state). When `run_id` is
`nothing`, no disk I/O happens — Phase 2 behaviour is preserved.

Phase 2 deliberately ignores `cfg.interventions` — broadcasts and
similar live interventions are wired in Phase 5 alongside the
dashboard.
"""
function run_simulation(
        cfg::WorldConfig;
        brain::AbstractBrain = NullBrain(),
        n_steps::Integer = cfg.max_ticks,
        run_id::Union{Nothing, AbstractString} = nothing
)::DataFrame
    if run_id === nothing
        model = initialize_world(cfg; brain = brain)
        return _run_loop(model, brain, n_steps)
    end

    return open_store(run_id) do db
        record_meta(db, cfg; run_id = run_id)
        model = initialize_world(cfg; brain = brain)
        model.store = db
        return _run_loop(model, brain, n_steps; db = db)
    end
end

function _run_loop(
        model, brain::AbstractBrain, n_steps::Integer;
        db::Union{Nothing, DuckDB.DB} = nothing
)::DataFrame
    ticks = Int[]
    ids = Int[]
    opinions = Float64[]
    persona_ids = String[]
    _collect_tick!(ticks, ids, opinions, persona_ids, model, 0)
    db === nothing || flush_tick!(db, model, 0)
    for t in 1:n_steps
        # ADR-0002 §1: batch all reflections at the *start* of a tick
        # so every agent sees the same brain outputs regardless of
        # update order. NullBrain returns an empty Dict — no LLM cost.
        batch_reflect!(brain, model; tick = t)
        step!(model, 1)
        _collect_tick!(ticks, ids, opinions, persona_ids, model, t)
        db === nothing || flush_tick!(db, model, t)
    end
    return DataFrame(
        tick = ticks,
        agent_id = ids,
        opinion = opinions,
        persona_id = persona_ids
    )
end

function _collect_tick!(ticks, ids, opinions, persona_ids, model, t::Integer)
    tt = Int(t)
    for a in allagents(model)
        push!(ticks, tt)
        push!(ids, Int(a.id))
        push!(opinions, a.opinion)
        push!(persona_ids, String(a.persona_id))
    end
    return nothing
end
