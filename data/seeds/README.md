# Seed schema

Each `*.json` here is a scenario consumed by `load_world(path)` and parses
into a [`WorldConfig`](../../src/io.jl). One file per scenario. Field names
mirror the structs verbatim.

The smallest valid scenario is in [`baseline.json`](baseline.json); read
that first, then use this file as the field-by-field reference.

## Top-level (`WorldConfig`)

| Field                  | Type                          | Notes |
|------------------------|-------------------------------|-------|
| `name`                 | string                        | Scenario id; matches the filename stem. |
| `description`          | string                        | Human-readable summary. |
| `seed`                 | integer                       | RNG seed; the *only* source of stochasticity under `EIDOLON_LLM_MODE=mock`. |
| `n_agents`             | integer                       | Total agents to instantiate. |
| `max_ticks`            | integer                       | Simulation horizon. |
| `personas`             | array of `AgentPersona`       | Embedded — no cross-file refs in v1. |
| `persona_distribution` | object: `persona_id → float`  | Fractions; **must sum to 1.0**. Every key must match a `personas[].id`. |
| `topology`             | `GraphTopology`               | See below. |
| `llm_model`            | string                        | Pinned model name. Recorded in the run's `meta` table (DuckDB). |
| `interventions`        | array of `Intervention`       | Optional; empty array if none. |

## `AgentPersona`

| Field                | Type             | Validation |
|----------------------|------------------|------------|
| `id`                 | string           | Unique within the scenario. |
| `description`        | string           | Natural-language; fed to the LLM brain. |
| `opinion_prior_mean` | float            | ∈ [0, 1]. Sampled `Normal(mean, std)` then clamped. |
| `opinion_prior_std`  | float            | ≥ 0. |
| `confidence_radius`  | float            | ε in HK (ADR-0001). ∈ (0, 1]. |
| `update_weight`      | float            | α in HK (ADR-0001). ∈ [0, 1]. |
| `memory_capacity`    | integer          | ≥ 0. Memory entries retained; older ones evicted. |
| `tags`               | array of string  | Free-form labels (demographic, role). |

## `GraphTopology`

| Field    | Type    | Notes |
|----------|---------|-------|
| `kind`   | string  | v1: `"watts_strogatz"`. |
| `params` | object  | Kind-specific. For `watts_strogatz`: `k` (int, even) and `beta` (float ∈ [0, 1]). |

## `Intervention`

| Field    | Type    | Notes |
|----------|---------|-------|
| `tick`   | integer | Tick at which the event fires. |
| `kind`   | string  | v1: `"broadcast"`. |
| `payload` | object | Kind-specific. For `broadcast`: `message` (string) and `intensity` (float ∈ [0, 1]). |

## Validation rules (enforced by the parser, not the JSON schema)

- `sum(values(persona_distribution)) ≈ 1.0` (tolerance 1e-9).
- `keys(persona_distribution) ⊆ {p.id for p in personas}`.
- `0 < confidence_radius ≤ 1`; `0 ≤ update_weight ≤ 1`.
- For `watts_strogatz`: `k` even, `0 ≤ k < n_agents`.
- `interventions[].tick ∈ [0, max_ticks]`.

## Minimal scenario

```json
{
    "name": "tiny",
    "description": "One persona, no interventions.",
    "seed": 0,
    "n_agents": 10,
    "max_ticks": 5,
    "personas": [
        {
            "id": "neutral",
            "description": "No prior allegiance.",
            "opinion_prior_mean": 0.5,
            "opinion_prior_std": 0.1,
            "confidence_radius": 0.3,
            "update_weight": 0.3,
            "memory_capacity": 4,
            "tags": []
        }
    ],
    "persona_distribution": { "neutral": 1.0 },
    "topology": {
        "kind": "watts_strogatz",
        "params": { "k": 4, "beta": 0.1 }
    },
    "llm_model": "claude-haiku-4-5-20251001",
    "interventions": []
}
```
