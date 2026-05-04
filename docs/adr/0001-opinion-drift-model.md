# ADR-0001: Opinion-drift model

## Status
Proposed — 2026-05-04

## Context

Phase 2 of the roadmap introduces `agent_step!`, the per-tick rule that
updates an agent's opinion given its neighbours and persona. The choice
of update rule shapes:

- the `EidolonAgent` struct (continuous vs discrete opinion, scalar vs
  vector, what state must persist between ticks)
- what "convergence" means in Phase 4 (one consensus, multiple clusters,
  polarised attractors, oscillation)
- whether the LLM brain (Phase 3) layers on top of, replaces, or
  short-circuits the social dynamics

The candidates considered:

| Model | State | Behaviour | Cost | Verdict |
|-------|-------|-----------|------|---------|
| **Hegselmann–Krause (HK)** — bounded confidence | continuous in `[0,1]` | clusters/polarisation depending on confidence radius ε | cheap, deterministic | **chosen baseline** |
| **DeGroot** — linear weighted average | continuous | always converges to consensus on a connected graph | cheap | rejected — no interesting macro phenomena |
| **Voter model** — copy random neighbour | discrete | absorbs into consensus, but slowly | cheap | rejected — discrete state ill-fits LLM-generated personas |
| **Friedkin–Johnsen** — DeGroot + stubbornness anchor | continuous | persistent disagreement | cheap | viable; deferred as a future variant |
| **Pure LLM-mediated** — ask the brain at each tick | text/numeric, free-form | most expressive | tens of dollars per run, non-deterministic | rejected as the *only* mechanism — keep as a perturbation, not the substrate |

## Decision

Use **Hegselmann–Krause as the deterministic substrate**, with the LLM
brain (Phase 3) acting as a **perturbation channel** layered on top.

Concretely, `agent_step!` follows this shape:

```julia
function agent_step!(agent, model)
    neighbours = nearby_agents(agent, model)
    in_confidence = filter(n -> abs(n.opinion - agent.opinion) ≤ agent.ε, neighbours)
    hk_target = isempty(in_confidence) ? agent.opinion : mean(n.opinion for n in in_confidence)
    Δ_brain   = brain_perturbation(agent, model)   # 0.0 under MockBrain
    agent.opinion = clamp(
        agent.α * hk_target + (1 - agent.α) * agent.opinion + Δ_brain,
        0.0, 1.0,
    )
    return agent
end
```

Implications for the data model:

- `EidolonAgent` carries `opinion::Float64`, `ε::Float64` (confidence
  radius), `α::Float64` (HK weight), `persona::AgentPersona`, plus
  `memory` (used only by `brain_perturbation`).
- `ε` and `α` come from the persona — different personas drift
  differently — so the schema in Phase 1 must include them.
- The opinion is **scalar for now**. Promote to `SVector{N,Float64}` if
  multi-issue dynamics become a research question; the HK update
  generalises trivially.

## Consequences

**Upsides**

- Phase 2 can ship and produce trajectories *without any LLM call*. CI
  with `EIDOLON_LLM_MODE=mock` exercises the full social-dynamics layer.
- `MockBrain` becomes trivial — `brain_perturbation` returns `0.0`. This
  is exactly the property required by Phase 3's done-criterion ("identical
  run reproducible under mock mode given the same RNG seed").
- HK has a large existing literature on cluster formation, so Phase 4's
  macro metrics (cluster count, polarisation index, time-to-stability)
  have established definitions.
- GlobalSensitivity.jl over `(ε, α)` and persona-distribution parameters
  is tractable because the substrate is deterministic.

**Downsides / follow-ups**

- HK with constant ε is well-studied — the *novelty* of Eidolon depends
  almost entirely on `brain_perturbation`. Phase 3 must produce
  perturbations that are non-trivial yet bounded (so HK convergence
  proofs still loosely apply).
- `Δ_brain` magnitude needs a cap (e.g. `|Δ_brain| ≤ Δ_max`) or a single
  rogue LLM response can flip an agent across the [0,1] range and wreck
  reproducibility. Track as a Phase 3 sub-task.
- Scalar opinion is a real limitation; revisit before Phase 4 if
  sensitivity results are uninterpretable.
- Friedkin–Johnsen ("stubbornness") may end up being the more honest
  model once personas have ideological anchors. Not chosen now, but
  watch for it.

## Open follow-ups

- ADR for `Δ_brain` bound and how the brain encodes its perturbation
  (signed scalar? distribution sample? text-to-number parser?) — covered
  by ADR-0002 when written.
- Add the chosen `ε`, `α` ranges to the baseline seed JSON in Phase 1.
