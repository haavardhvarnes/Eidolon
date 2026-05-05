# Phase 2.5 — DuckDB persistence per ADR-0004.
#
# One DuckDB file per run, at `runs/<run_id>/store.duckdb` by default.
# Tests redirect the location via the `EIDOLON_RUNS_ROOT` environment
# variable so they can sandbox each run into a `mktempdir`.

using Agents: allagents
using DBInterface
using Dates: DateTime, now, UTC
using DuckDB
using JSON
using Printf: @sprintf
using TOML

const _SCHEMA_VERSION = 1

const _SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        schema_version  INTEGER  NOT NULL,
        run_id          VARCHAR  NOT NULL,
        created_at      TIMESTAMP NOT NULL,
        seed            BIGINT   NOT NULL,
        llm_model       VARCHAR  NOT NULL,
        eidolon_version VARCHAR  NOT NULL,
        julia_version   VARCHAR  NOT NULL,
        config_json     VARCHAR  NOT NULL,
        manifest_toml   VARCHAR  NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS personas (
        id                 VARCHAR PRIMARY KEY,
        description        VARCHAR NOT NULL,
        opinion_prior_mean DOUBLE  NOT NULL,
        opinion_prior_std  DOUBLE  NOT NULL,
        confidence_radius  DOUBLE  NOT NULL,
        update_weight      DOUBLE  NOT NULL,
        memory_capacity    INTEGER NOT NULL,
        tags_json          VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trajectory (
        tick       INTEGER NOT NULL,
        agent_id   INTEGER NOT NULL,
        opinion    DOUBLE  NOT NULL,
        persona_id VARCHAR NOT NULL REFERENCES personas(id),
        PRIMARY KEY (tick, agent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory (
        tick        INTEGER NOT NULL,
        agent_id    INTEGER NOT NULL,
        entry_index INTEGER NOT NULL,
        content     VARCHAR NOT NULL,
        PRIMARY KEY (tick, agent_id, entry_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        tick           INTEGER NOT NULL,
        agent_id       INTEGER NOT NULL,
        model          VARCHAR NOT NULL,
        template       VARCHAR NOT NULL,
        prompt         VARCHAR NOT NULL,
        response       VARCHAR,
        delta_raw      DOUBLE,
        delta_clamped  DOUBLE,
        status         VARCHAR NOT NULL,
        latency_ms     INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        tick    INTEGER NOT NULL,
        kind    VARCHAR NOT NULL,
        payload VARCHAR NOT NULL
    )
    """
)

const _PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

const _EIDOLON_VERSION = let
    project_path = joinpath(_PROJECT_ROOT, "Project.toml")
    String(TOML.parsefile(project_path)["version"])
end

"""
    runs_root() -> String

Resolve the directory under which `runs/<run_id>/store.duckdb` files
live. Defaults to `runs/` at the project root; override via the
`EIDOLON_RUNS_ROOT` environment variable so tests can sandbox each run
into a `mktempdir`.
"""
function runs_root()
    haskey(ENV, "EIDOLON_RUNS_ROOT") && return ENV["EIDOLON_RUNS_ROOT"]
    return joinpath(_PROJECT_ROOT, "runs")
end

"""
    store_path(run_id) -> String

Path to the DuckDB file for `run_id` under the configured runs root.
"""
function store_path(run_id::AbstractString)
    return joinpath(runs_root(), String(run_id), "store.duckdb")
end

function _init_schema!(db::DuckDB.DB)
    for sql in _SCHEMA_SQL
        DBInterface.execute(db, sql)
    end
    return db
end

"""
    open_store(f, run_id)

Open (or create) `runs/<run_id>/store.duckdb`, ensure the ADR-0004 §2
schema exists (`CREATE TABLE IF NOT EXISTS` for every table — re-opening
is idempotent), pass the [`DuckDB.DB`](@ref) handle to `f`, and close
the database on exit. Standard `do`-block idiom:

```julia
open_store(run_id) do db
    record_meta(db, cfg; run_id = run_id)
    flush_tick!(db, model, 0)
end
```
"""
function open_store(f::Function, run_id::AbstractString)
    path = store_path(run_id)
    mkpath(dirname(path))
    db = DBInterface.connect(DuckDB.DB, path)
    try
        _init_schema!(db)
        return f(db)
    finally
        DBInterface.close!(db)
    end
end

# --- Config / persona round-trip -------------------------------------

function _persona_to_dict(p::AgentPersona)
    return Dict{String, Any}(
        "id" => p.id,
        "description" => p.description,
        "opinion_prior_mean" => p.opinion_prior_mean,
        "opinion_prior_std" => p.opinion_prior_std,
        "confidence_radius" => p.confidence_radius,
        "update_weight" => p.update_weight,
        "memory_capacity" => p.memory_capacity,
        "tags" => p.tags
    )
end

function _topology_to_dict(t::GraphTopology)
    return Dict{String, Any}("kind" => t.kind, "params" => t.params)
end

function _intervention_to_dict(iv::Intervention)
    return Dict{String, Any}(
        "tick" => iv.tick,
        "kind" => iv.kind,
        "payload" => iv.payload
    )
end

function _config_to_dict(cfg::WorldConfig)
    return Dict{String, Any}(
        "name" => cfg.name,
        "description" => cfg.description,
        "seed" => cfg.seed,
        "n_agents" => cfg.n_agents,
        "max_ticks" => cfg.max_ticks,
        "personas" => [_persona_to_dict(p) for p in cfg.personas],
        "persona_distribution" => cfg.persona_distribution,
        "topology" => _topology_to_dict(cfg.topology),
        "llm_model" => cfg.llm_model,
        "interventions" => [_intervention_to_dict(iv) for iv in cfg.interventions]
    )
end

function _read_manifest_toml()
    path = joinpath(_PROJECT_ROOT, "Manifest.toml")
    return isfile(path) ? read(path, String) : ""
end

# --- Writes ----------------------------------------------------------

"""
    record_meta(db, cfg::WorldConfig; seed = cfg.seed, run_id::AbstractString)

Write the single `meta` row and bulk-insert `cfg.personas` for the run.
The `meta` row pins schema version, RNG seed, LLM model, this package's
version, the running Julia version, the resolved `WorldConfig` as JSON,
and a snapshot of `Manifest.toml` — together the reproducibility
contract from ADR-0004 §2.
"""
function record_meta(
        db::DuckDB.DB,
        cfg::WorldConfig;
        seed::Integer = cfg.seed,
        run_id::AbstractString
)
    config_json = JSON.json(_config_to_dict(cfg))
    manifest = _read_manifest_toml()
    DBInterface.execute(
        db,
        """
        INSERT INTO meta
            (schema_version, run_id, created_at, seed, llm_model,
             eidolon_version, julia_version, config_json, manifest_toml)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            Int32(_SCHEMA_VERSION),
            String(run_id),
            now(UTC),
            Int64(seed),
            cfg.llm_model,
            _EIDOLON_VERSION,
            string(VERSION),
            config_json,
            manifest
        )
    )

    appender = DuckDB.Appender(db, "personas")
    try
        for p in cfg.personas
            DuckDB.append(appender, p.id)
            DuckDB.append(appender, p.description)
            DuckDB.append(appender, p.opinion_prior_mean)
            DuckDB.append(appender, p.opinion_prior_std)
            DuckDB.append(appender, p.confidence_radius)
            DuckDB.append(appender, p.update_weight)
            DuckDB.append(appender, Int32(p.memory_capacity))
            DuckDB.append(appender, JSON.json(p.tags))
            DuckDB.end_row(appender)
        end
        DuckDB.flush(appender)
    finally
        DuckDB.close(appender)
    end
    return nothing
end

"""
    flush_tick!(db, model, tick::Integer)

Bulk-insert one row per agent into `trajectory` for `tick` (one batched
insert via `DuckDB.Appender`, not one INSERT per row — ADR-0004 §3).
"""
function flush_tick!(db::DuckDB.DB, model, tick::Integer)
    t = Int32(tick)
    appender = DuckDB.Appender(db, "trajectory")
    try
        for a in allagents(model)
            DuckDB.append(appender, t)
            DuckDB.append(appender, Int32(a.id))
            DuckDB.append(appender, Float64(a.opinion))
            DuckDB.append(appender, String(a.persona_id))
            DuckDB.end_row(appender)
        end
        DuckDB.flush(appender)
    finally
        DuckDB.close(appender)
    end
    return nothing
end

# --- Read-only inspection (ADR-0004 §"Open follow-ups") --------------

_fmt_opinion(x::Real) = @sprintf("%.3f", x)
_fmt_opinion(::Missing) = "NA"

function _scalar(db::DuckDB.DB, sql::AbstractString, params = ())
    return only(DBInterface.execute(db, sql, params))
end

function _print_meta(io::IO, db::DuckDB.DB)
    println(io, "Meta")
    rows = collect(DBInterface.execute(
        db,
        """
        SELECT created_at, seed, llm_model, eidolon_version, julia_version
        FROM meta
        """
    ))
    if isempty(rows)
        println(io, "  (no meta row recorded)")
        return nothing
    end
    m = first(rows)
    println(io, "  created_at:      ", m.created_at)
    println(io, "  seed:            ", m.seed)
    println(io, "  llm_model:       ", m.llm_model)
    println(io, "  eidolon_version: ", m.eidolon_version)
    println(io, "  julia_version:   ", m.julia_version)
    return nothing
end

function _print_personas(io::IO, db::DuckDB.DB)
    rows = collect(DBInterface.execute(db, "SELECT id FROM personas ORDER BY id"))
    ids = [String(r.id) for r in rows]
    label = "Personas (" * string(length(ids)) * "):"
    if isempty(ids)
        println(io, label, " (none)")
    else
        println(io, label, " ", join(ids, ", "))
    end
    return nothing
end

function _print_trajectory(io::IO, db::DuckDB.DB)
    println(io, "Trajectory")
    summary = _scalar(
        db,
        """
        SELECT COUNT(*) AS n,
               MIN(tick) AS first_tick,
               MAX(tick) AS last_tick,
               COUNT(DISTINCT agent_id) AS agents
        FROM trajectory
        """
    )
    if summary.n == 0
        println(io, "  rows:    0")
        return nothing
    end
    println(io, "  rows:    ", summary.n)
    println(io, "  ticks:   ", summary.first_tick, "..", summary.last_tick)
    println(io, "  agents:  ", summary.agents)
    for (label, t) in (("first", summary.first_tick), ("last", summary.last_tick))
        stats = _scalar(
            db,
            """
            SELECT AVG(opinion) AS mean,
                   MIN(opinion) AS lo,
                   MAX(opinion) AS hi
            FROM trajectory
            WHERE tick = ?
            """,
            (t,)
        )
        println(
            io,
            "  tick ", t, " (", label, "): mean=", _fmt_opinion(stats.mean),
            " min=", _fmt_opinion(stats.lo),
            " max=", _fmt_opinion(stats.hi)
        )
    end
    return nothing
end

function _print_aux_counts(io::IO, db::DuckDB.DB)
    for table in ("memory", "transcripts", "events")
        n = _scalar(db, "SELECT COUNT(*) AS n FROM $table").n
        println(io, rpad(uppercasefirst(table) * ":", 13), n, " rows")
    end
    return nothing
end

"""
    dump_run(run_id::AbstractString; io::IO = stdout) -> Nothing

Print a one-screen, human-readable summary of `runs/<run_id>/store.duckdb`
to `io`: header, `meta` row, persona ids, trajectory shape + opinion
stats at the first and last tick, and row counts for `memory`,
`transcripts`, and `events`. Read-only — opens the DuckDB file with
`access_mode = READ_ONLY` so the call never takes a write lock and
never mutates schema. Throws `ArgumentError` if no store exists at the
resolved path.
"""
function dump_run(run_id::AbstractString; io::IO = stdout)
    path = store_path(run_id)
    if !isfile(path)
        throw(ArgumentError(
            "dump_run: no DuckDB store for run_id \"$(run_id)\" at $path " *
            "(run a simulation with this run_id first, or check EIDOLON_RUNS_ROOT)",
        ))
    end
    db = DBInterface.connect(DuckDB.DB, path; readonly = true)
    try
        println(io, "Run:   ", run_id)
        println(io, "Store: ", path)
        println(io)
        _print_meta(io, db)
        println(io)
        _print_personas(io, db)
        println(io)
        _print_trajectory(io, db)
        println(io)
        _print_aux_counts(io, db)
    finally
        DBInterface.close!(db)
    end
    return nothing
end

"""
    list_runs(; root::AbstractString = runs_root()) -> Vector{String}

Return the sorted ids of runs persisted under `root` — i.e. each
subdirectory of `root` that contains a `store.duckdb` file. Returns an
empty vector if `root` does not exist.
"""
function list_runs(; root::AbstractString = runs_root())
    isdir(root) || return String[]
    ids = String[]
    for entry in readdir(root)
        if isfile(joinpath(root, entry, "store.duckdb"))
            push!(ids, entry)
        end
    end
    sort!(ids)
    return ids
end
