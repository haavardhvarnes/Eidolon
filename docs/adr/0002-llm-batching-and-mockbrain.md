# ADR-0002: LLM batching strategy, brain interface, and Δ bound

## Status
Proposed — 2026-05-04

## Context

Phase 3 of the roadmap delivers `brain.jl` — the LLM-driven cognitive layer.
ADR-0001 commits to a Hegselmann–Krause substrate plus a per-tick
perturbation `Δ_brain`, with `MockBrain` returning zero so that CI and
sensitivity sweeps stay deterministic. This ADR fixes:

1. The brain *interface* — what `agent_step!` calls and what types it
   exchanges.
2. The MockBrain / LiveBrain split, including additional flavours useful
   for testing.
3. How per-tick LLM calls are *batched* and rate-limited.
4. The bound on `Δ_brain` (deferred from ADR-0001) — both prompted to the
   model and enforced in code.
5. Retry, cost-cap, and transcript-persistence policy.

Constraints driving the design:

- **Reproducibility under `EIDOLON_LLM_MODE=mock`** (ADR-0001 done-criterion
  for Phase 3).
- **Cost containment** — a 1 000-agent × 100-tick run with one LLM call
  per agent per tick is 100 k calls. Without batching and a kill switch
  this is unaffordable.
- **No silent reproducibility breakage** — a rogue LLM response that
  returns `Δ = 0.97` would flip an agent across `[0, 1]` and ruin
  comparisons across runs.
- **PromptingTools.jl** is the chosen LLM dependency (`Project.toml`).
  We should use its `aiextract` / `airetry!` rather than reinvent.

## Decision

### 1. Interface — two-phase per tick

```julia
abstract type AbstractBrain end

"""
    BrainOutput

What the brain returns for one agent at one tick.
- `delta`: bounded opinion perturbation, see ADR-0001.
- `new_memory`: text to append to the agent's memory ring.
- `reasoning`: free-text rationale; persisted to transcripts only.
"""
struct BrainOutput
    delta::Float64
    new_memory::String
    reasoning::String
end

# Phase A — collect all reflections at the *start* of a tick (batched, async).
# Returns a Dict{AgentID, BrainOutput} that lives for the duration of the tick.
batch_reflect!(brain::AbstractBrain, model) -> Dict{Int,BrainOutput}

# Phase B — agent_step! reads the precomputed entry (sync, free).
brain_perturbation(brain::AbstractBrain, agent, model) -> Float64
```

`brain_perturbation` is the hook called from `agent_step!` (ADR-0001). It
looks up `model.brain_outputs[agent.id].delta`, applies the Δ cap (see §4),
and returns the scalar. The "two-phase per tick" pattern is the standard
ABM read-only-snapshot idiom: every agent in tick `t` sees the same brain
outputs, regardless of update order.

### 2. Brain hierarchy

| Concrete type            | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `NullBrain`              | Returns `Δ = 0.0` for all agents. Default under `EIDOLON_LLM_MODE=mock`. Matches ADR-0001 reproducibility contract. |
| `RandomBrain(rng, σ)`    | Returns `Δ ∼ Normal(0, σ)` (clamped). For testing the perturbation pipeline without LLM cost. |
| `ReplayBrain(transcripts)` | Replays a prior run's `transcripts` table. Regression tests. |
| `LiveBrain(model, sem, retry, cap)` | Calls PromptingTools. Production. |

`MockBrain` from ADR-0001 is the conceptual umbrella for the first three.
Code-level, it is just `AbstractBrain`.

### 3. Batching — async with bounded concurrency

`batch_reflect!(::LiveBrain, model)`:

1. Build N prompts (one per agent) from `agent.persona`, `agent.memory`,
   and the tick's neighbour snapshot.
2. Dispatch via `asyncmap(prompt -> aiextract(prompt, BrainOutput); ntasks = brain.sem.limit)`.
   Default concurrency limit: 16. PromptingTools handles HTTP keep-alive.
3. Wrap each call with `airetry!` (exponential backoff, max 3 attempts).
4. On terminal failure for any agent, that agent's output falls back to
   `BrainOutput(0.0, "", "<retry-exhausted>")` — the run continues. A
   warning row goes to the DuckDB `transcripts` table with `status = "failed"`.
5. Each successful (or failed) call writes one row to `transcripts`
   (prompt, response, latency_ms, model, status).

`MockBrain` variants are synchronous and have no rate limiting.

### 4. Δ bound — two-layer enforcement

- **Prompted bound**: the reflection template explicitly tells the model
  `delta` must lie in `[-Δ_max, +Δ_max]`.
- **Enforced bound**: `clamp(raw, -Δ_max, +Δ_max)` is applied in
  `brain_perturbation` regardless of what the model returns. The clamped
  vs raw value is recorded in the transcript so excessive clamping is
  visible.

`Δ_max` for v1: **`0.05`** (5% of opinion range per tick). Override via
the `EIDOLON_BRAIN_DELTA_CAP` env var. Move to a `WorldConfig`/persona
field in v2 once we have evidence about reasonable values.

Rationale: with `α ≤ 0.5` (typical persona), the maximum HK pull per tick
is ~`0.5 · max_neighbour_delta`. Capping the LLM perturbation at the same
order of magnitude keeps the LLM channel from dominating the social
dynamics.

### 5. Cost guardrail

`LiveBrain` reads `EIDOLON_MAX_LLM_CALLS` (default `5_000` in dev,
unset = unlimited in production). A tick-level counter is incremented per
dispatched call; on overflow, the run aborts with a typed error
`LLMBudgetExceeded` and the partially-written DuckDB store is left intact
for inspection.

### 6. Retry policy

`airetry!` with: 3 attempts, exponential backoff (0.5 s, 2 s, 8 s),
retry only on transient errors (HTTP 429, 5xx, network timeouts). Schema
violations (`aiextract` parse failure) are *not* retried — they are
treated as terminal and counted against the failure budget; if more than
5% of agents in a tick fail to parse, the run aborts (`SchemaDriftError`).

### 7. Prompt template

A single PromptingTools template `:agent_reflection_v1` registered in
`src/brain.jl` at module load. The template references:

- `{{persona_description}}` — from `agent.persona.description`
- `{{recent_memory}}` — newest-first slice of `agent.memory`, truncated to fit
- `{{neighbour_opinions}}` — small numeric array, anonymised (no agent ids)
- `{{global_event}}` — most recent intervention payload, if any

The template is versioned in its name; bumping it requires a new ADR
(any change to prompts can shift sensitivity results materially).

### 8. Transcript persistence

Every call (success, failure, retry) writes one row to the DuckDB
`transcripts` table (Storage section in CLAUDE.md). Schema:

| column        | type     |
|---------------|----------|
| `tick`        | INTEGER  |
| `agent_id`    | INTEGER  |
| `model`       | VARCHAR  |
| `template`    | VARCHAR  |
| `prompt`      | VARCHAR  |
| `response`    | VARCHAR  |
| `delta_raw`   | DOUBLE   |
| `delta_clamped` | DOUBLE |
| `status`      | VARCHAR  |  -- "ok" | "retried" | "failed" | "schema_error"
| `latency_ms`  | INTEGER  |

This is the only artefact that makes a `LiveBrain` run replayable by
`ReplayBrain` later — and the only thing that makes sensitivity-analysis
results reproducible across days.

## Consequences

**Upsides**

- The two-phase tick model lets `agent_step!` stay synchronous and
  cheap; only `batch_reflect!` is async. Easy to reason about and to
  test deterministically.
- The Δ cap is enforced even if a future model misbehaves or a prompt
  regression sneaks through — code is the floor, not the prompt.
- `ReplayBrain` from prior transcripts gives us a "golden master" testing
  channel for free, on top of `NullBrain` and `RandomBrain`.
- Cost guardrail is a hard limit — `--dangerously-skip-permissions`
  cannot accidentally burn the user's API budget.

**Downsides / follow-ups**

- The two-phase pattern means agents within one tick **cannot** observe
  each other's brain outputs; the brain layer is read-only within a tick.
  Sequential-influence dynamics (e.g. a charismatic agent shifting its
  immediate neighbour mid-tick) are lost. Acceptable trade-off for v1;
  revisit if results look wrong.
- Async batching with a hard concurrency cap can still hit provider rate
  limits during sweeps. Add per-minute throttling (`Δt` between batch
  releases) only when this becomes a real problem.
- Schema-drift threshold (5% in §6) is heuristic. Tune after observing
  the first 10 live runs.
- `Δ_max = 0.05` is a guess. The first non-trivial run should sweep
  `Δ_max ∈ {0.02, 0.05, 0.10}` and pick the value where the sensitivity
  results stay stable.

## Open follow-ups

- Decide where `Δ_max` lives in v2 (`WorldConfig` field vs per-persona).
  Likely per-persona — different personas should be more or less
  LLM-suggestible. Defer until ADR-0001 is moved from Proposed to
  Accepted with empirical data.
- Decide whether `template` versioning sits in `src/brain.jl` constants
  or in a `templates/` directory. Defer.
- Memory-eviction policy is referenced in the schema (`memory_capacity`)
  but the eviction rule (FIFO? LRU? salience-weighted?) belongs to a
  later ADR.
