using Eidolon
using DataFrames
using DBInterface
using DuckDB
using Test

const STORE_BASELINE_PATH = joinpath(@__DIR__, "..", "data", "seeds", "baseline.json")

# Tables enumerated in ADR-0004 §2 — sorted alphabetically because the
# `information_schema.tables` query below sorts by `table_name`.
const _ADR0004_TABLES = [
    "events", "memory", "meta", "personas", "trajectory", "transcripts"]

function _table_names(db)
    res = DBInterface.execute(
        db,
        "SELECT table_name FROM information_schema.tables " *
        "WHERE table_schema = 'main' ORDER BY table_name"
    )
    return [String(row.table_name) for row in res]
end

_count(db, table) = only(DBInterface.execute(db, "SELECT COUNT(*) AS n FROM $table")).n

@testset "store" begin
    @testset "open_store creates the ADR-0004 schema (six tables)" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                open_store("smoke") do db
                    @test _table_names(db) == _ADR0004_TABLES
                end
                @test isfile(joinpath(tmp, "smoke", "store.duckdb"))
            end
        end
    end

    @testset "record_meta writes one row with schema_version = 1" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                cfg = load_world(STORE_BASELINE_PATH)
                open_store("v1") do db
                    record_meta(db, cfg; run_id = "v1")
                    @test _count(db, "meta") == 1
                    row = only(DBInterface.execute(
                        db,
                        "SELECT schema_version, run_id, seed, llm_model FROM meta"
                    ))
                    @test row.schema_version == 1
                    @test row.run_id == "v1"
                    @test row.seed == cfg.seed
                    @test row.llm_model == cfg.llm_model
                    @test _count(db, "personas") == length(cfg.personas)
                end
            end
        end
    end

    @testset "re-opening preserves schema and inserted rows (idempotency)" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                cfg = load_world(STORE_BASELINE_PATH)
                open_store("idem") do db
                    record_meta(db, cfg; run_id = "idem")
                end
                # Second open_store on the same run_id must not duplicate
                # the schema or wipe the previously inserted rows.
                open_store("idem") do db
                    @test _table_names(db) == _ADR0004_TABLES
                    @test _count(db, "meta") == 1
                    @test _count(db, "personas") == length(cfg.personas)
                end
            end
        end
    end

    @testset "run_simulation persists trajectory matching the returned DataFrame" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                cfg = load_world(STORE_BASELINE_PATH)
                df = run_simulation(cfg; run_id = "baseline")

                @test df isa DataFrame
                @test nrow(df) == cfg.n_agents * (cfg.max_ticks + 1)

                open_store("baseline") do db
                    @test _count(db, "trajectory") == cfg.n_agents * (cfg.max_ticks + 1)
                    @test _count(db, "meta") == 1
                    @test _count(db, "personas") == length(cfg.personas)

                    meta_row = only(DBInterface.execute(
                        db,
                        "SELECT seed, llm_model FROM meta"
                    ))
                    @test meta_row.seed == cfg.seed
                    @test meta_row.llm_model == cfg.llm_model

                    persisted = DataFrame(
                        DBInterface.execute(
                        db,
                        "SELECT tick, agent_id, opinion, persona_id FROM trajectory"
                    )
                    )
                    sort!(persisted, [:tick, :agent_id])
                    expected = sort(df, [:tick, :agent_id])
                    @test persisted.tick == expected.tick
                    @test persisted.agent_id == expected.agent_id
                    @test persisted.opinion == expected.opinion
                    @test persisted.persona_id == expected.persona_id
                end
            end
        end
    end
end
