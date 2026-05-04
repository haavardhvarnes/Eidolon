# ADR-0004: Storage backend — DuckDB, one file per run

## Status
Proposed — 2026-05-04

## Context

Eidolon writes a non-trivial amount of state per run: per-tick trajectories,
agent memories, LLM transcripts (ADR-0002), broadcast/intervention events,
and a meta record covering RNG seed and pinned model. Phase 4 sensitivity
sweeps then need to query *across* many such runs. CLAUDE.md already
sketches a `runs/<run-id>/` layout and names DuckDB as the chosen store;
this ADR formalises that decision, the schema, the write strategy, and
the cross-run query approach.

Alternatives considered, with reasons rejected:

| Option | Why not |
|--------|---------|
| **Flat CSV + JSONL** | Cheapest to start, but cross-run queries need bespoke globbing; no schema enforcement; no efficient column scans. |
| **SQLite** | Mature, embedded, but row-oriented — Phase 4 columnar scans across millions of trajectory rows are slow. |
| **Postgres** | Overkill for a single-machine sim; adds an out-of-process service to start before any sim. |
| **Parquet only** | Excellent for analysis but append-during-run is awkward — Parquet is write-once-per-file. We'd end up with one Parquet per tick, fragmented. |
| **DuckDB + Parquet sidecar** | DuckDB can already query Parquet, and exporting a duplicate is redundant. Defer until cross-run scans become slow. |

## Decision

### 1. One DuckDB file per run

`runs/<run-id>/store.duckdb` is the canonical artefact for a run. It
contains everything needed to reproduce, analyse, or audit that run.
The `runs/` directory is gitignored (`.gitignore`); reproducibility comes
from the seed + pinned model + manifest stored *inside* the file, not from
checking the file in.

Rationale for per-run files (not a single shared DB):

- **Isolation**: a crash mid-tick can corrupt at most that run.
- **Parallel sweeps**: each run is a writer, no contention.
- **Trivially shippable**: copy one file to share an experiment.
- **Easy GC**: `rm -rf runs/<run-id>/` reclaims everything.

### 2. Schema (v1)

```sql
-- Recorded once at run start.
CREATE TABLE meta (
    schema_version  INTEGER  NOT NULL,    -- this ADR = 1
    run_id          VARCHAR  NOT NULL,
    created_at      TIMESTAMP NOT NULL,
    seed            BIGINT   NOT NULL,
    llm_model       VARCHAR  NOT NULL,    -- pinned by WorldConfig
    eidolon_version VARCHAR  NOT NULL,    -- Project.toml version
    julia_version   VARCHAR  NOT NULL,
    config_json     VARCHAR  NOT NULL,    -- the resolved WorldConfig as JSON
    manifest_toml   VARCHAR  NOT NULL     -- snapshot of Manifest.toml
);

-- Resolved persona table (denormalised from WorldConfig, for query convenience).
CREATE TABLE personas (
    id                 VARCHAR PRIMARY KEY,
    description        VARCHAR NOT NULL,
    opinion_prior_mean DOUBLE  NOT NULL,
    opinion_prior_std  DOUBLE  NOT NULL,
    confidence_radius  DOUBLE  NOT NULL,
    update_weight      DOUBLE  NOT NULL,
    memory_capacity    INTEGER NOT NULL,
    tags_json          VARCHAR NOT NULL    -- JSON array
);

-- Per-tick agent state. Hottest table for Phase 4.
CREATE TABLE trajectory (
    tick       INTEGER NOT NULL,
    agent_id   INTEGER NOT NULL,
    opinion    DOUBLE  NOT NULL,
    persona_id VARCHAR NOT NULL REFERENCES personas(id),
    PRIMARY KEY (tick, agent_id)
);

-- Append-only per-agent memory.
CREATE TABLE memory (
    tick        INTEGER NOT NULL,
    agent_id    INTEGER NOT NULL,
    entry_index INTEGER NOT NULL,    -- position within the agent's ring
    content     VARCHAR NOT NULL,
    PRIMARY KEY (tick, agent_id, entry_index)
);

-- LLM call log; schema fixed by ADR-0002 §8.
CREATE TABLE transcripts (
    tick           INTEGER NOT NULL,
    agent_id       INTEGER NOT NULL,
    model          VARCHAR NOT NULL,
    template       VARCHAR NOT NULL,
    prompt         VARCHAR NOT NULL,
    response       VARCHAR,
    delta_raw      DOUBLE,
    delta_clamped  DOUBLE,
    status         VARCHAR NOT NULL,    -- ok | retried | failed | schema_error
    latency_ms     INTEGER
);

-- Broadcasts and other interventions.
CREATE TABLE events (
    tick    INTEGER NOT NULL,
    kind    VARCHAR NOT NULL,
    payload VARCHAR NOT NULL    -- JSON
);
```

`schema_version` lives in `meta` so a future migration script can detect
old files. Bumping the schema requires a new ADR.

### 3. Write strategy

- **Connection lifecycle.** One `DuckDB.DB` per run, opened by
  `open_store(run_id)` (returns a closure-friendly handle) and closed at
  the end of the run. Standard Julia `do`-block idiom:

  ```julia
  open_store(run_id) do db
      record_meta(db, config, seed)
      for tick in 1:max_ticks
          step!(model)
          flush_tick!(db, model, tick)
      end
  end
  ```

- **Batching.** `flush_tick!` writes one `INSERT` per affected table
  per tick, not one row at a time. For 100 agents × 100 ticks the
  per-tick batch is small enough that this is just performance hygiene,
  not a correctness concern.

- **Transcripts** are written by the brain layer directly — they are
  not part of `flush_tick!`, since some calls happen async during
  `batch_reflect!` (ADR-0002).

- **WAL.** DuckDB's `*.wal` is gitignored. On a crash the WAL replays
  on next open; if the user wants a clean re-run, deleting both
  `store.duckdb` and `store.duckdb.wal` is the documented recovery.

### 4. Cross-run analysis (Phase 4)

A `read_run(run_id)` helper opens a single run and returns DataFrames
for the relevant tables. For sweeps:

```julia
# Phase 4 sketch: open a single analysis DB, ATTACH each run, return
# trajectory + meta as one DataFrame.
function load_trajectories(run_ids::Vector{String})
    db = DBInterface.connect(DuckDB.DB, ":memory:")
    for (i, id) in enumerate(run_ids)
        path = joinpath("runs", id, "store.duckdb")
        DBInterface.execute(db, "ATTACH '$path' AS r$i (READ_ONLY)")
    end
    sql = join(("SELECT '$id' AS run_id, * FROM r$i.trajectory"
                for (i, id) in enumerate(run_ids)), " UNION ALL ")
    return DataFrame(DBInterface.execute(db, sql))
end
```

DuckDB's `ATTACH` lets us treat N run-files as one logical store without
copying. Only promote to a Parquet sidecar if scans across thousands of
runs become slow — premature today.

### 5. Concurrency

Single Julia process per run = single writer + transactional reads
within that process. DuckDB's MVCC handles this. No explicit locking
required. Concurrent *runs* live in separate files, so no contention.

## Consequences

**Upsides**

- One-file-per-run gives trivially deletable, shippable, parallel-safe
  artefacts.
- Columnar storage makes Phase 4 sweeps over `trajectory` fast even at
  millions of rows.
- DuckDB reads CSV/Parquet/JSON natively, so importing third-party data
  (e.g. real-world opinion polls for validation) needs zero glue.
- Schema versioning via `meta.schema_version` lets us migrate later
  without leaving old runs unreadable.

**Downsides / follow-ups**

- DuckDB files are not human-readable. Add a `dump_run(run_id)` helper
  that prints summary stats so debugging doesn't always require a SQL prompt.
- Storing the full prompt/response per LLM call makes `transcripts` the
  biggest table by far. For huge runs we may need to compress text or
  archive transcripts to gzipped JSONL out-of-band — defer until size
  becomes a problem.
- `manifest_toml` is duplicated across every run file. Cheap (KBs), but
  a thousand-run sweep duplicates ~MB. Acceptable.
- `events.payload` and `personas.tags_json` are stored as JSON strings.
  DuckDB has native JSON support but we're keeping the schema portable
  for now. Promote to native JSON columns if query ergonomics suffer.

## Open follow-ups

- `dump_run(run_id)` text summariser and a small CLI (`scripts/inspect_run.jl`).
- Decide naming convention for `run_id` — recommend `<scenario>-<timestamp>-<short_hash>` so listing `runs/` is informative.
- A `migrate_runs.jl` skeleton (no-op until the schema actually changes)
  that reads `meta.schema_version` and bails on unknown versions.
- Cross-run analysis API: define `load_trajectories` / `load_transcripts`
  signatures formally when Phase 4 starts. Today this ADR just commits
  to the underlying mechanism (`ATTACH` + `UNION ALL`).
