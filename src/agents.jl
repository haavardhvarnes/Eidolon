# Phase 2 — @agent EidolonAgent and the per-tick opinion update.

using Agents
using Statistics: mean

"""
    EidolonAgent

Agent type for the swarm. Lives on a `GraphSpace` (Watts–Strogatz
small-world graph) and updates its scalar opinion via the
Hegselmann–Krause rule (ADR-0001), with an optional brain perturbation
channel layered on top (ADR-0002).

Fields:
- `opinion::Float64` — scalar opinion in `[0, 1]`.
- `ε::Float64`       — HK confidence radius, copied from the persona.
- `α::Float64`       — HK update weight, copied from the persona.
- `persona_id::String` — id of the originating `AgentPersona`.
- `memory::Vector{String}` — append-only memory; populated by the
  brain in Phase 3, kept empty in Phase 2.
"""
@agent struct EidolonAgent(GraphAgent)
    opinion::Float64
    ε::Float64
    α::Float64
    persona_id::String
    memory::Vector{String}
end

"""
    agent_step!(agent, model)

Hegselmann–Krause bounded-confidence update with the brain
perturbation channel (ADR-0001 §Decision). Reads `model.brain` and
delegates to [`brain_perturbation`](@ref); under `NullBrain` the
perturbation is `0.0` and the dynamics reduce to vanilla HK.
"""
function agent_step!(agent, model)
    neighbours = collect(nearby_agents(agent, model))
    in_confidence = filter(n -> abs(n.opinion - agent.opinion) ≤ agent.ε, neighbours)
    hk_target = isempty(in_confidence) ? agent.opinion :
                mean(n.opinion for n in in_confidence)
    Δ = brain_perturbation(model.brain, agent, model)
    agent.opinion = clamp(
        agent.α * hk_target + (1.0 - agent.α) * agent.opinion + Δ,
        0.0,
        1.0
    )
    return agent
end
