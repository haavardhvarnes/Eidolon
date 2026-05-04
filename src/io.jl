# Phase 1 — Data ingestion. Schemas and the `load_world` parser. The
# JSON-side schema reference lives in `data/seeds/README.md`.

using JSON

"""
    AgentPersona

Template for a kind of agent. Multiple `EidolonAgent`s share a persona.

Fields with HK semantics (see ADR-0001):
- `confidence_radius` — ε in HK, max neighbour-opinion distance considered.
- `update_weight`     — α in HK, how much an agent moves toward the mean.
"""
struct AgentPersona
    id::String
    description::String
    opinion_prior_mean::Float64
    opinion_prior_std::Float64
    confidence_radius::Float64
    update_weight::Float64
    memory_capacity::Int
    tags::Vector{String}
end

"""
    GraphTopology

Discriminated by `kind`; valid kinds for v1: `"watts_strogatz"`.
"""
struct GraphTopology
    kind::String
    params::Dict{String, Any}
end

"""
    Intervention

A scheduled event applied at simulation tick `tick`. Kinds for v1:
`"broadcast"` (inject a message into every agent's memory).
"""
struct Intervention
    tick::Int
    kind::String
    payload::Dict{String, Any}
end

"""
    WorldConfig

Top-level scenario description. One-to-one with `data/seeds/<name>.json`.
"""
struct WorldConfig
    name::String
    description::String
    seed::Int
    n_agents::Int
    max_ticks::Int
    personas::Vector{AgentPersona}
    persona_distribution::Dict{String, Float64}
    topology::GraphTopology
    llm_model::String
    interventions::Vector{Intervention}
end

"""
    SchemaError(msg)

Raised by `load_world` when a seed file violates the schema or any of
the validation rules listed in `data/seeds/README.md`. The message is
intended to point an author straight at the offending field.
"""
struct SchemaError <: Exception
    msg::String
end

Base.showerror(io::IO, e::SchemaError) = print(io, "SchemaError: ", e.msg)

# --- Field accessors -------------------------------------------------

function _require_field(d::AbstractDict, key::AbstractString, ctx::AbstractString)
    haskey(d, key) || throw(SchemaError("$ctx: missing required field \"$key\""))
    return d[key]
end

function _as_string(x, field::AbstractString, ctx::AbstractString)
    x isa AbstractString || throw(SchemaError(
        "$ctx: field \"$field\" must be a string, got $(typeof(x))",
    ))
    return String(x)
end

function _as_int(x, field::AbstractString, ctx::AbstractString)
    if x isa Integer && !(x isa Bool)
        return Int(x)
    elseif x isa AbstractFloat && isinteger(x)
        return Int(x)
    else
        throw(SchemaError(
            "$ctx: field \"$field\" must be an integer, got $(repr(x))",
        ))
    end
end

function _as_float(x, field::AbstractString, ctx::AbstractString)
    (x isa Real && !(x isa Bool)) || throw(SchemaError(
        "$ctx: field \"$field\" must be a number, got $(typeof(x))",
    ))
    return Float64(x)
end

function _as_object(x, field::AbstractString, ctx::AbstractString)
    x isa AbstractDict || throw(SchemaError(
        "$ctx: field \"$field\" must be an object, got $(typeof(x))",
    ))
    return x
end

function _as_array(x, field::AbstractString, ctx::AbstractString)
    x isa AbstractVector || throw(SchemaError(
        "$ctx: field \"$field\" must be an array, got $(typeof(x))",
    ))
    return x
end

# --- Hydration -------------------------------------------------------

function _hydrate_persona(raw, idx::Int)
    ctx = "personas[$idx]"
    raw isa AbstractDict ||
        throw(SchemaError("$ctx: must be an object, got $(typeof(raw))"))
    id = _as_string(_require_field(raw, "id", ctx), "id", ctx)
    description = _as_string(_require_field(raw, "description", ctx), "description", ctx)
    mean = _as_float(_require_field(raw, "opinion_prior_mean", ctx), "opinion_prior_mean", ctx)
    std = _as_float(_require_field(raw, "opinion_prior_std", ctx), "opinion_prior_std", ctx)
    cr = _as_float(_require_field(raw, "confidence_radius", ctx), "confidence_radius", ctx)
    uw = _as_float(_require_field(raw, "update_weight", ctx), "update_weight", ctx)
    capacity = _as_int(_require_field(raw, "memory_capacity", ctx), "memory_capacity", ctx)
    tags_raw = _as_array(_require_field(raw, "tags", ctx), "tags", ctx)
    tags = [_as_string(t, "tags[$i]", ctx) for (i, t) in enumerate(tags_raw)]

    (0.0 < cr <= 1.0) || throw(SchemaError(
        "$ctx (id=\"$id\"): confidence_radius must satisfy 0 < confidence_radius ≤ 1, got $cr",
    ))
    (0.0 <= uw <= 1.0) || throw(SchemaError(
        "$ctx (id=\"$id\"): update_weight must satisfy 0 ≤ update_weight ≤ 1, got $uw",
    ))

    return AgentPersona(id, description, mean, std, cr, uw, capacity, tags)
end

function _hydrate_topology(raw, n_agents::Int)
    ctx = "topology"
    raw isa AbstractDict ||
        throw(SchemaError("$ctx: must be an object, got $(typeof(raw))"))
    kind = _as_string(_require_field(raw, "kind", ctx), "kind", ctx)
    params_raw = _as_object(_require_field(raw, "params", ctx), "params", ctx)
    params = Dict{String, Any}(String(k) => v for (k, v) in params_raw)

    if kind == "watts_strogatz"
        haskey(params, "k") || throw(SchemaError(
            "$ctx: watts_strogatz requires \"k\" in params",
        ))
        haskey(params, "beta") || throw(SchemaError(
            "$ctx: watts_strogatz requires \"beta\" in params",
        ))
        k = _as_int(params["k"], "params.k", ctx)
        (0 <= k < n_agents) || throw(SchemaError(
            "$ctx: watts_strogatz requires 0 ≤ k < n_agents, got k=$k, n_agents=$n_agents",
        ))
        iseven(k) || throw(SchemaError(
            "$ctx: watts_strogatz requires k to be even, got k=$k",
        ))
    else
        throw(SchemaError(
            "$ctx: unsupported topology kind \"$kind\" (v1 supports: watts_strogatz)",
        ))
    end

    return GraphTopology(kind, params)
end

function _hydrate_intervention(raw, idx::Int, max_ticks::Int)
    ctx = "interventions[$idx]"
    raw isa AbstractDict ||
        throw(SchemaError("$ctx: must be an object, got $(typeof(raw))"))
    tick = _as_int(_require_field(raw, "tick", ctx), "tick", ctx)
    kind = _as_string(_require_field(raw, "kind", ctx), "kind", ctx)
    payload_raw = _as_object(_require_field(raw, "payload", ctx), "payload", ctx)
    payload = Dict{String, Any}(String(k) => v for (k, v) in payload_raw)

    (0 <= tick <= max_ticks) || throw(SchemaError(
        "$ctx: tick must be in [0, max_ticks=$max_ticks], got $tick",
    ))

    return Intervention(tick, kind, payload)
end

function _validate_persona_distribution(
        pd::Dict{String, Float64},
        personas::Vector{AgentPersona}
)
    persona_ids = Set(p.id for p in personas)
    for k in keys(pd)
        k in persona_ids || throw(SchemaError(
            "persona_distribution: key \"$k\" does not match any persona id " *
            "(known: $(sort(collect(persona_ids))))",
        ))
    end
    total = sum(values(pd))
    isapprox(total, 1.0; atol = 1.0e-9) || throw(SchemaError(
        "persona_distribution: fractions must sum to 1.0 (tolerance 1e-9), got $total",
    ))
    return nothing
end

function _hydrate_world(raw)::WorldConfig
    raw isa AbstractDict || throw(SchemaError(
        "top-level JSON must be an object, got $(typeof(raw))",
    ))
    ctx = "WorldConfig"
    name = _as_string(_require_field(raw, "name", ctx), "name", ctx)
    description = _as_string(_require_field(raw, "description", ctx), "description", ctx)
    seed = _as_int(_require_field(raw, "seed", ctx), "seed", ctx)
    n_agents = _as_int(_require_field(raw, "n_agents", ctx), "n_agents", ctx)
    max_ticks = _as_int(_require_field(raw, "max_ticks", ctx), "max_ticks", ctx)

    personas_raw = _as_array(_require_field(raw, "personas", ctx), "personas", ctx)
    personas = [_hydrate_persona(p, i) for (i, p) in enumerate(personas_raw)]

    pd_raw = _as_object(_require_field(raw, "persona_distribution", ctx), "persona_distribution", ctx)
    persona_distribution = Dict{String, Float64}(
        String(k) => _as_float(v, "persona_distribution[\"$k\"]", ctx)
    for (k, v) in pd_raw
    )

    topology = _hydrate_topology(_require_field(raw, "topology", ctx), n_agents)
    llm_model = _as_string(_require_field(raw, "llm_model", ctx), "llm_model", ctx)

    interv_raw = _as_array(_require_field(raw, "interventions", ctx), "interventions", ctx)
    interventions = [_hydrate_intervention(it, i, max_ticks)
                     for (i, it) in enumerate(interv_raw)]

    _validate_persona_distribution(persona_distribution, personas)

    return WorldConfig(
        name,
        description,
        seed,
        n_agents,
        max_ticks,
        personas,
        persona_distribution,
        topology,
        llm_model,
        interventions
    )
end

"""
    load_world(path)::WorldConfig

Parse a seed file at `path` into a fully validated `WorldConfig`.
Validation rules are listed in `data/seeds/README.md`; violations
throw [`SchemaError`](@ref) with a message identifying the offending
field.
"""
function load_world(path::AbstractString)::WorldConfig
    isfile(path) || throw(SchemaError("seed file not found: $path"))
    raw = JSON.parsefile(path)
    return _hydrate_world(raw)
end
