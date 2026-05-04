# Phase 3 — LLM logic. MockBrain / LiveBrain interface lands here.
# Phase 2 only ships the minimal stable surface from ADR-0002 so that
# `agent_step!` has a perturbation hook; everything else (BrainOutput,
# batch_reflect!, transcripts, retry policy) is deferred to Phase 3.

"""
    AbstractBrain

Supertype for all cognitive backends (ADR-0002 §1). Concrete subtypes
arriving in Phase 3: `RandomBrain`, `ReplayBrain`, `LiveBrain`. Phase
2 only ships [`NullBrain`](@ref).
"""
abstract type AbstractBrain end

"""
    NullBrain

The reproducibility-preserving brain: returns Δ = 0 for every agent
on every tick. Default under `EIDOLON_LLM_MODE=mock`. Honours the
ADR-0001 done-criterion that runs are bit-reproducible given the same
RNG seed.
"""
struct NullBrain <: AbstractBrain end

"""
    brain_perturbation(brain, agent, model) -> Float64

Per-tick opinion perturbation Δ called from [`agent_step!`](@ref)
(ADR-0002 §1). Phase 2 only defines the `NullBrain` dispatch; the
batched-LLM path lands in Phase 3.
"""
brain_perturbation(::NullBrain, agent, model) = 0.0
