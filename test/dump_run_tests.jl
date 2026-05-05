using Eidolon
using Test

const DUMP_BASELINE_PATH = joinpath(@__DIR__, "..", "data", "seeds", "baseline.json")

@testset "dump_run / list_runs" begin
    @testset "populated store summarises meta, personas, agent count" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                cfg = load_world(DUMP_BASELINE_PATH)
                run_simulation(cfg; run_id = "dump-baseline")

                buf = IOBuffer()
                dump_run("dump-baseline"; io = buf)
                out = String(take!(buf))

                @test !isempty(out)
                @test occursin("dump-baseline", out)
                @test occursin(string(cfg.seed), out)
                @test occursin(cfg.llm_model, out)
                for p in cfg.personas
                    @test occursin(p.id, out)
                end
                # `agents:` line should report the configured agent count.
                @test occursin("agents:  $(cfg.n_agents)", out)
            end
        end
    end

    @testset "missing run_id throws with run_id and path in message" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                expected_path = joinpath(tmp, "ghost", "store.duckdb")
                err = try
                    dump_run("ghost"; io = devnull)
                    nothing
                catch e
                    e
                end
                @test err !== nothing
                msg = sprint(showerror, err)
                @test occursin("ghost", msg)
                @test occursin(expected_path, msg)
                @test !isfile(expected_path)
                # Containing directory must NOT be auto-created.
                @test !isdir(joinpath(tmp, "ghost"))
            end
        end
    end

    @testset "empty store (schema only, no rows) prints without crashing" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                open_store("empty") do _db
                    # Schema initialised, no rows inserted.
                end

                buf = IOBuffer()
                dump_run("empty"; io = buf)
                out = String(take!(buf))

                @test !isempty(out)
                @test occursin("empty", out)
                @test occursin("rows:    0", out)
                @test occursin("Memory:", out)
                @test occursin("Transcripts:", out)
                @test occursin("Events:", out)
            end
        end
    end

    @testset "list_runs returns sorted ids; ignores plain dirs" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                # Order of creation is intentionally non-alphabetical.
                for id in ("charlie", "alpha", "bravo")
                    open_store(id) do _db
                    end
                end
                # A directory without a store.duckdb must be skipped.
                mkpath(joinpath(tmp, "no-store"))

                @test list_runs() == ["alpha", "bravo", "charlie"]
            end
        end
    end

    @testset "list_runs returns empty vector when root is missing" begin
        mktempdir() do tmp
            missing_root = joinpath(tmp, "does-not-exist")
            @test !isdir(missing_root)
            @test list_runs(root = missing_root) == String[]
        end
    end
end
