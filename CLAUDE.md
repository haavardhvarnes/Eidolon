# Eidolon.jl

Predictive swarm-intelligence simulator: an agent-based model where each
agent's reasoning is driven by an LLM "brain", coupled to a small-world
social network. The goal is to study macro-level outcomes (opinion
convergence, polarisation, cascades) and identify which agent personas
drive them.

> Repository: https://github.com/haavardhvarnes/Eidolon
> Eidolon.jl is a Julia port of the [**MiroFish**](https://github.com/666ghj/MiroFish) project.

## Project Structure

```
Eidolon/
├── .claude/                   # Claude Code instructions for this project
├── data/seeds/                # JSON seed files (initial world & personas)
├── scripts/
│   └── run_simulation.jl      # Entry point to execute the swarm
├── src/
│   ├── Eidolon.jl             # Main module (includes other files)
│   ├── agents.jl              # @agent definitions & personality logic
│   ├── world.jl               # GraphSpace & world initialisation
│   ├── brain.jl               # LLM logic (PromptingTools.jl)
│   ├── io.jl                  # JSON parsing & data collection
│   └── dashboard.jl           # Stipple-based control panel
├── docs/adr/                  # Architecture Decision Records (one file per decision)
├── runs/                      # Per-run artefacts: DuckDB file, transcripts, manifest
├── test/runtests.jl
├── Project.toml
├── Manifest.toml              # auto-generated; do not hand-edit
└── .JuliaFormatter.toml       # style = "sciml"
```

## Environment

- Activate before any command: `julia --project=.` (or `Pkg.activate(".")`
  then `Pkg.instantiate()` inside a REPL).
- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Run sim:   `julia --project=. scripts/run_simulation.jl`
- LLM mode:  `EIDOLON_LLM_MODE=mock|live` (default `mock` so CI is free
  and deterministic).

## Coding Standards

Follow the SciML Style Guide. Full conventions live in `~/.claude/CLAUDE.md`;
the load-bearing rules for this repo:

- `lower_snake_case` for functions/variables, `CamelCase` for types/modules
- 4-space indent, LF line endings, `1.0` not `1.`
- Format with Runic.jl using `.JuliaFormatter.toml` (`style = "sciml"`)
- Prefer immutable structs; mark mutating functions with `!`
- Type-stable, generic across array & numeric types — use `similar(A)`
- JSON.jl (not JSON3.jl, which is deprecated)

## Preferred Packages

| Concern         | Package                          |
|-----------------|----------------------------------|
| ABM             | Agents.jl                        |
| Graphs          | Graphs.jl (Watts–Strogatz)       |
| LLM             | PromptingTools.jl                |
| HTTP            | HTTP.jl                          |
| Data I/O        | JSON.jl, CSV.jl, Tables.jl       |
| Analysis        | DataFrames.jl, Distributions.jl  |
| Sensitivity     | GlobalSensitivity.jl             |
| Storage         | DuckDB.jl (primary), SQLite.jl (fallback) |
| UI              | Stipple.jl, StipplePlotly        |

## Storage

State that survives a run lives in **DuckDB** — a single embedded file
per run. Chosen because it is columnar (fast Phase 4 sweeps), reads
CSV/Parquet/JSON natively (zero-friction migration from flat files),
and pairs cleanly with DataFrames.jl. See ADR-0004.

Schema (one DuckDB file per run, at `runs/<run-id>/store.duckdb`):

| Table          | Contents                                        |
|----------------|-------------------------------------------------|
| `meta`         | run-id, RNG seed, package manifest, LLM model   |
| `personas`     | resolved `AgentPersona` rows                    |
| `trajectory`   | per-tick agent state (id, tick, opinion, …)    |
| `memory`       | per-agent append-only memory entries            |
| `transcripts`  | full LLM prompt/response log (one row per call) |
| `events`       | broadcasts and other interventions              |

Deferred until proven necessary:

- **Graph store** (Memgraph/Neo4j) — start with an in-memory
  `MetaGraph` from Graphs.jl. Promote only if GraphRAG-style retrieval
  becomes the LLM-side bottleneck.
- **Vector store** — start with cosine similarity over an in-process
  embedding matrix; HNSW.jl if it gets slow; external vector DB only
  if the simulator goes multi-process.

This is also the reproducibility contract: pinned model + seed +
manifest + full transcripts means a sensitivity result can be
re-derived months later, even though the LLM itself is
non-deterministic.

## Architecture Decision Records

Non-trivial design choices live in `docs/adr/NNNN-title.md`, numbered
sequentially. Write an ADR before committing to a decision that would be
expensive to reverse — opinion-drift model, batching strategy, dashboard
transport, etc. Skip ADRs for obvious or easily reversible choices.

Template:

```markdown
# ADR-NNNN: <title>

## Status
Proposed | Accepted | Superseded by ADR-MMMM — YYYY-MM-DD

## Context
What forced the decision? What constraints / alternatives matter?

## Decision
The choice, stated as a clear sentence.

## Consequences
+ Upsides
− Downsides and follow-ups
```

Initial ADRs to write before Phase 2 / Phase 3:

- ADR-0001: Opinion-drift model (Hegselmann–Krause vs DeGroot vs voter)
- ADR-0002: LLM batching strategy and `MockBrain` interface
- ADR-0003: Stipple ↔ Agents.jl event-loop bridge
- ADR-0004: Storage backend (DuckDB single-file vs flat files vs Postgres)

## Plan Hierarchy

Three layers, in order of permanence:

1. **Roadmap** — the phases below. Durable, project-wide direction.
2. **Session plans** — `~/.claude/plans/<slug>.md`. Ephemeral, scoped
   to one Claude Code session.
3. **ADRs** — `docs/adr/NNNN-*.md`. Capture irreversible decisions
   with their rationale.

Promote a roadmap phase to its own `docs/plans/phase-N.md` only when
active work on it outgrows the bullets here (≳7 sub-tasks, multiple
unknowns, or a parallel collaborator). Don't pre-create them.

## Development Roadmap

Mark items `[x]` as they complete. Each phase has a definition-of-done
that should pass before moving on.

### Phase 0 — Standards & Environment
- [ ] Create `Project.toml` with pinned `julia` compat
- [ ] Add `.JuliaFormatter.toml` (`style = "sciml"`)
- [ ] CI workflow running `Pkg.test()` with `EIDOLON_LLM_MODE=mock`
- **Done when:** `julia --project=. -e 'using Pkg; Pkg.instantiate()'` succeeds clean.

### Phase 1 — Data Ingestion (`io.jl`)
- [ ] Define `WorldConfig` and `AgentPersona` structs + JSON schema
- [ ] `load_world(path)` hydrates structs directly from JSON
- [ ] At least one baseline seed in `data/seeds/baseline.json`
- [ ] `open_store(run_id)` initialises `runs/<run-id>/store.duckdb` with the
      schema in the Storage section and writes the `meta`/`personas` rows
- **Done when:** `load_world("data/seeds/baseline.json")` populates a config,
  `open_store("test")` produces a queryable DuckDB file, and both are covered by tests.

### Phase 2 — Digital Sandbox (`agents.jl`, `world.jl`)
- [ ] `@agent EidolonAgent` with `opinion`, `memory`, `persona`
- [ ] `GraphSpace` initialised from a Watts–Strogatz small-world graph
- [ ] `agent_step!` covering social influence + opinion drift
      (model: bounded-confidence / Hegselmann–Krause — see ADR-001)
- **Done when:** a 100-agent, 50-step run completes without LLM and produces a trajectory.

### Phase 3 — Cognitive Layer (`brain.jl`)
- [ ] PromptingTools.jl templates for reflection and dialogue
- [ ] `MockBrain` and `LiveBrain` behind a common interface
- [ ] `batch_reflect!` — async batched LLM calls per tick, with retry/rate-limit handling
- **Done when:** identical run reproducible under `mock` mode given the same RNG seed.

### Phase 4 — Macro Prediction
- [ ] Query trajectories across runs via DuckDB → DataFrames.jl
- [ ] Parameter sweep harness (grid before Sobol), one DuckDB file per run
- [ ] GlobalSensitivity.jl over persona parameters with mock brain only
- **Done when:** sensitivity report identifies top-3 personas by influence on a chosen macro metric.

### Phase 5 — Web Control Panel (`dashboard.jl`)
- [ ] Stipple UI with start/pause/reset toggles
- [ ] StipplePlotly network visualisation
- [ ] "Broadcast" intervention — inject global events into agent memories
- **Done when:** a non-Julia user can launch a sim and broadcast an event from the browser.

> Phase 4 is intentionally promoted ahead of the dashboard: headless
> analysis is the scientific deliverable; the UI is a demo.
