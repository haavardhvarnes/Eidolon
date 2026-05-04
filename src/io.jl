# Phase 1 — Data ingestion. Schemas defined here; parser (load_world)
# lands in the Phase 1 implementation session.

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
