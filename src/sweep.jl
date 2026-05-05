# Phase 4 — Cross-run aggregation and parameter sweep harness.
# ADR-0004 §4 defines the cross-run read path (ATTACH + UNION ALL).

using DataFrames
using Dates: format, now
using DBInterface
using DuckDB
using JSON
using SHA

# --- run id generation -----------------------------------------------

"""
    auto_run_id(cfg::WorldConfig) -> String

Generate a human-readable, provenance-bearing run identifier.
Format: `"<cfg.name>-<YYYYmmdd-HHMMSS>-<8-hex-hash>"`.

The 8-character hex hash is `bytes2hex(SHA.sha1(cfg_json))[1:8]` where
`cfg_json` is the canonical JSON serialisation of `cfg`. Identical
configs produce the same hash component; the timestamp segment
distinguishes calls made more than one second apart.
"""
function auto_run_id(cfg::WorldConfig)
    cfg_json = JSON.json(_config_to_dict(cfg))
    hash_str = bytes2hex(SHA.sha1(cfg_json))[1:8]
    ts = format(now(), "yyyymmdd-HHMMSS")
    return "$(cfg.name)-$ts-$hash_str"
end

# --- helpers ---------------------------------------------------------

# Escape single quotes in a string for embedding in a SQL literal.
_esc_sql(s::AbstractString) = replace(String(s), "'" => "''")

function _trajectory_empty()
    return DataFrame(
        run_id = String[],
        tick = Int32[],
        agent_id = Int32[],
        opinion = Float64[],
        persona_id = String[]
    )
end

function _transcripts_empty()
    return DataFrame(
        run_id = String[],
        tick = Int32[],
        agent_id = Int32[],
        model = String[],
        template = String[],
        prompt = String[],
        response = String[],
        delta_raw = Float64[],
        delta_clamped = Float64[],
        status = String[],
        latency_ms = Int32[]
    )
end

# Attach the store for `id` at `root` to `db` under alias `alias`.
# Returns `true` on success; emits `@warn` and returns `false` if the
# file does not exist.
function _attach_store!(
        db::DuckDB.DB, id::AbstractString, alias::AbstractString,
        root::AbstractString, context::AbstractString
)
    path = joinpath(root, String(id), "store.duckdb")
    if !isfile(path)
        @warn "$context: no store for run_id \"$id\" at $path — skipping"
        return false
    end
    DBInterface.execute(db, "ATTACH '$(_esc_sql(path))' AS $alias (READ_ONLY)")
    return true
end

# --- load_trajectories -----------------------------------------------

"""
    load_trajectories(run_ids; root = runs_root()) -> DataFrame

Open an in-memory DuckDB instance, `ATTACH` each run's store in
`READ_ONLY` mode, and `UNION ALL` the `trajectory` tables into one
`DataFrame`. Columns: `run_id`, `tick`, `agent_id`, `opinion`,
`persona_id`.

Missing stores emit a one-line `@warn` and are skipped. Empty input
returns an empty `DataFrame` with the documented columns (no error).
"""
function load_trajectories(
        run_ids::AbstractVector{<:AbstractString};
        root::AbstractString = runs_root()
)
    isempty(run_ids) && return _trajectory_empty()

    db = DBInterface.connect(DuckDB.DB, ":memory:")
    try
        attached = Tuple{String, String}[]   # (alias, id)
        for (i, id) in enumerate(run_ids)
            alias = "r$i"
            if _attach_store!(db, id, alias, root, "load_trajectories")
                push!(attached, (alias, String(id)))
            end
        end

        isempty(attached) && return _trajectory_empty()

        parts = ["SELECT '$(_esc_sql(id))' AS run_id, " *
                 "tick, agent_id, opinion, persona_id FROM $alias.trajectory"
                 for (alias, id) in attached]
        return DataFrame(DBInterface.execute(db, join(parts, " UNION ALL ")))
    finally
        DBInterface.close!(db)
    end
end

# --- load_transcripts ------------------------------------------------

"""
    load_transcripts(run_ids; root = runs_root()) -> DataFrame

Same as [`load_trajectories`](@ref) but for the `transcripts` table.
Columns: `run_id`, `tick`, `agent_id`, `model`, `template`, `prompt`,
`response`, `delta_raw`, `delta_clamped`, `status`, `latency_ms`.

Missing stores emit a one-line `@warn` and are skipped. Empty input
returns an empty `DataFrame` with the documented columns (no error).
"""
function load_transcripts(
        run_ids::AbstractVector{<:AbstractString};
        root::AbstractString = runs_root()
)
    isempty(run_ids) && return _transcripts_empty()

    db = DBInterface.connect(DuckDB.DB, ":memory:")
    try
        attached = Tuple{String, String}[]
        for (i, id) in enumerate(run_ids)
            alias = "r$i"
            if _attach_store!(db, id, alias, root, "load_transcripts")
                push!(attached, (alias, String(id)))
            end
        end

        isempty(attached) && return _transcripts_empty()

        parts = ["SELECT '$(_esc_sql(id))' AS run_id, " *
                 "tick, agent_id, model, template, prompt, response, " *
                 "delta_raw, delta_clamped, status, latency_ms " *
                 "FROM $alias.transcripts"
                 for (alias, id) in attached]
        return DataFrame(DBInterface.execute(db, join(parts, " UNION ALL ")))
    finally
        DBInterface.close!(db)
    end
end

# --- expand_grid -----------------------------------------------------

"""
    expand_grid(spec::NamedTuple) -> Vector{NamedTuple}

Return the Cartesian product of the value vectors in `spec`. The result
is ordered lexicographically over `spec`'s field order: the first field
varies slowest (outermost loop), the last field varies fastest.

```julia
expand_grid((sigma = [0.01, 0.05], confidence_radius = [0.2, 0.4]))
# → 4 NamedTuples: (0.01,0.2), (0.01,0.4), (0.05,0.2), (0.05,0.4)
```

An empty value vector on any field returns an empty vector.
"""
function expand_grid(spec::NamedTuple)
    fields = keys(spec)
    isempty(fields) && return NamedTuple[]
    for k in fields
        isempty(spec[k]) && return NamedTuple[]
    end
    vectors = [collect(spec[k]) for k in fields]
    # Reverse so the first field is outermost (slowest-changing).
    # Iterators.product's first argument varies fastest.
    result = NamedTuple[]
    sizehint!(result, prod(length(v) for v in vectors))
    for combo in Iterators.product(reverse(vectors)...)
        push!(result, NamedTuple{fields}(reverse(combo)))
    end
    return result
end

# --- grid_sweep ------------------------------------------------------

"""
    grid_sweep(cfg_factory, grid; brain_factory, n_steps_factory)
        -> Vector{Tuple{NamedTuple, String}}

Execute one `run_simulation` per grid point and return `(point, run_id)`
pairs. Runs are sequential (v1 — no parallelism).

Arguments:
- `cfg_factory`      — `point -> WorldConfig`. Called once per point.
- `grid`             — `AbstractVector{<:NamedTuple}`, e.g. from [`expand_grid`](@ref).
- `brain_factory`    — `(cfg, point) -> AbstractBrain` (default: `NullBrain()`).
- `n_steps_factory`  — `cfg -> Integer` (default: `cfg.max_ticks`).
"""
function grid_sweep(
        cfg_factory,
        grid::AbstractVector{<:NamedTuple};
        brain_factory = (cfg, point) -> NullBrain(),
        n_steps_factory = cfg -> cfg.max_ticks
)
    result = Vector{Tuple{NamedTuple, String}}()
    sizehint!(result, length(grid))
    for point in grid
        cfg = cfg_factory(point)::WorldConfig
        run_id = auto_run_id(cfg)
        brain = brain_factory(cfg, point)
        run_simulation(cfg; brain = brain, run_id = run_id, n_steps = n_steps_factory(cfg))
        push!(result, (point, run_id))
    end
    return result
end
