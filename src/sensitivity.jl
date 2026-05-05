# Phase 4 — Sobol sensitivity analysis over persona parameters.
#
# Uses GlobalSensitivity.jl with NullBrain only (ADR-0002 mock
# requirement). No DuckDB stores are written during analysis; each
# run_simulation call is stateless from the storage perspective.

using DataFrames
using GlobalSensitivity
using Statistics: var

# Persona parameters supported as sensitivity dimensions.
const _SENSIBLE_PARAMS = Set([:confidence_radius, :update_weight,
                               :opinion_prior_mean, :opinion_prior_std])

"""
    SensitivityDimension(persona_id, param, lo, hi)

One axis in a Sobol sensitivity analysis: a `(persona_id, param)` pair
and its search interval `[lo, hi]`. Valid `param` symbols are
`:confidence_radius`, `:update_weight`, `:opinion_prior_mean`, and
`:opinion_prior_std`.
"""
struct SensitivityDimension
    persona_id::String
    param::Symbol
    lo::Float64
    hi::Float64
end

"""
    SensitivityResult

Return type of [`persona_sensitivity`](@ref).

Fields:
- `dimensions`   — the [`SensitivityDimension`](@ref) axes in input order.
- `S1`           — first-order Sobol indices, one per dimension.
- `ST`           — total-order Sobol indices, one per dimension.
- `metric_name`  — label identifying the macro metric (display only).
"""
struct SensitivityResult
    dimensions::Vector{SensitivityDimension}
    S1::Vector{Float64}
    ST::Vector{Float64}
    metric_name::String
end

# --- Built-in metrics ------------------------------------------------

"""
    opinion_variance_metric(df::DataFrame) -> Float64

Final-tick variance of agent opinions — a scalar measure of spread /
polarisation. Higher values mean more dispersed opinions at convergence.
"""
function opinion_variance_metric(df::DataFrame)
    final_tick = maximum(df.tick)
    final = filter(r -> r.tick == final_tick, df).opinion
    return var(final; corrected = false)
end

# --- Internal helpers ------------------------------------------------

function _validate_dims(
        cfg::WorldConfig,
        dims::AbstractVector{SensitivityDimension}
)
    isempty(dims) && throw(ArgumentError(
        "persona_sensitivity: dims must be non-empty"
    ))
    persona_ids = Set(p.id for p in cfg.personas)
    for d in dims
        d.persona_id in persona_ids || throw(ArgumentError(
            "SensitivityDimension: persona_id \"$(d.persona_id)\" not in cfg " *
            "(known: $(sort!(collect(persona_ids))))"
        ))
        d.param in _SENSIBLE_PARAMS || throw(ArgumentError(
            "SensitivityDimension: unsupported param :$(d.param) " *
            "(supported: $(join(sort!([String(p) for p in _SENSIBLE_PARAMS]), ", ")))"
        ))
        d.lo < d.hi || throw(ArgumentError(
            "SensitivityDimension(\"$(d.persona_id)\", :$(d.param)): " *
            "lo=$(d.lo) must be strictly less than hi=$(d.hi)"
        ))
    end
    return nothing
end

# Construct a new WorldConfig with per-persona parameter overrides from
# the flat parameter vector `x` (one entry per dim, in order).
function _apply_dims(
        cfg::WorldConfig,
        dims::AbstractVector{SensitivityDimension},
        x::AbstractVector
)
    overrides = Dict{String, Dict{Symbol, Float64}}()
    for (i, dim) in enumerate(dims)
        d = get!(overrides, dim.persona_id, Dict{Symbol, Float64}())
        d[dim.param] = Float64(x[i])
    end

    new_personas = map(cfg.personas) do p
        haskey(overrides, p.id) || return p
        ov = overrides[p.id]
        AgentPersona(
            p.id,
            p.description,
            get(ov, :opinion_prior_mean, p.opinion_prior_mean),
            get(ov, :opinion_prior_std, p.opinion_prior_std),
            # Clamp to valid ranges so a Sobol sample at a boundary never
            # violates schema constraints (ADR-0001 §Consequences).
            clamp(get(ov, :confidence_radius, p.confidence_radius), 1.0e-6, 1.0),
            clamp(get(ov, :update_weight, p.update_weight), 0.0, 1.0),
            p.memory_capacity,
            p.tags
        )
    end

    return WorldConfig(
        cfg.name,
        cfg.description,
        cfg.seed,
        cfg.n_agents,
        cfg.max_ticks,
        new_personas,
        cfg.persona_distribution,
        cfg.topology,
        cfg.llm_model,
        cfg.interventions
    )
end

# --- persona_sensitivity ---------------------------------------------

"""
    persona_sensitivity(cfg_template, dims;
        metric          = opinion_variance_metric,
        metric_name     = "opinion_variance",
        n_samples       = 512,
        n_steps         = nothing,
    ) -> SensitivityResult

Run a Sobol first- and total-order sensitivity analysis over the persona
parameters described by `dims`.

`NullBrain` is used throughout so no LLM calls are made. No DuckDB
stores are written — simulations are run in-memory only.

Arguments:
- `cfg_template` — base `WorldConfig`; parameters outside `dims` are
                   held at their template values.
- `dims`         — `AbstractVector{SensitivityDimension}` specifying
                   which `(persona_id, param)` axes to vary and their
                   `[lo, hi]` bounds. Must be non-empty.
- `metric`       — `df::DataFrame -> Float64` called on the trajectory
                   returned by each `run_simulation`. Default:
                   [`opinion_variance_metric`](@ref).
- `metric_name`  — string label stored in the result (display only).
- `n_samples`    — base sample count `N` passed to `GlobalSensitivity.Sobol`.
                   Total function evaluations = `N × (d + 2)` where
                   `d = length(dims)`.
- `n_steps`      — ticks per run; defaults to `cfg_template.max_ticks`.
"""
function persona_sensitivity(
        cfg_template::WorldConfig,
        dims::AbstractVector{SensitivityDimension};
        metric::Function = opinion_variance_metric,
        metric_name::AbstractString = "opinion_variance",
        n_samples::Integer = 512,
        n_steps::Union{Nothing, Integer} = nothing
)::SensitivityResult
    _validate_dims(cfg_template, dims)
    steps = n_steps === nothing ? cfg_template.max_ticks : Int(n_steps)
    bounds = [[d.lo, d.hi] for d in dims]

    f = function (x::AbstractVector)
        cfg = _apply_dims(cfg_template, collect(dims), x)
        df = run_simulation(cfg; n_steps = steps)
        return metric(df)
    end

    raw = gsa(f, Sobol(), bounds; samples = Int(n_samples))

    # For scalar output gsa returns S1/ST as 1×d matrices; flatten to
    # a plain Vector to match the SensitivityResult contract.
    s1 = vec(raw.S1)
    st = vec(raw.ST)

    return SensitivityResult(collect(dims), s1, st, String(metric_name))
end

# --- top_personas ----------------------------------------------------

"""
    top_personas(result::SensitivityResult; n::Integer = 3)
        -> Vector{NamedTuple}

Aggregate total-order Sobol indices by persona (summing across all
dimensions that belong to the same persona) and return the top-`n`
personas sorted descending by aggregate sensitivity.

Each element of the returned vector is a NamedTuple
`(persona_id::String, total_sensitivity::Float64)`.
"""
function top_personas(result::SensitivityResult; n::Integer = 3)
    scores = Dict{String, Float64}()
    for (dim, st) in zip(result.dimensions, result.ST)
        scores[dim.persona_id] = get(scores, dim.persona_id, 0.0) + st
    end
    sorted = sort!(collect(scores); by = last, rev = true)
    k = min(Int(n), length(sorted))
    return [(persona_id = p, total_sensitivity = s) for (p, s) in sorted[1:k]]
end
