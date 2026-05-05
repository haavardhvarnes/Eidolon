using Eidolon
using DataFrames
using DBInterface
using DuckDB
using Test

const REPLAY_BASELINE_PATH = joinpath(@__DIR__, "..", "data", "seeds", "baseline.json")

# Stub dispatch factory: returns a constant delta for every agent.
function _replay_stub(delta::Real)
    d = Float64(delta)
    return (brain, aid,
        vars) -> (
        prompt = "p$aid",
        response = "r$aid",
        output = BrainOutput(d, "", "")
    )
end

@testset "replay" begin
    @testset "equivalence: ReplayBrain reproduces LiveBrain trajectory" begin
        cfg = load_world(REPLAY_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp, "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain_live = live_brain(cfg; dispatch = _replay_stub(0.01), max_concurrency = 8)
                T1 = run_simulation(cfg; brain = brain_live, run_id = "eq_src", n_steps = 3)

                rb = replay_brain("eq_src")
                T2 = run_simulation(cfg; brain = rb, n_steps = 3)

                sort!(T1, [:tick, :agent_id])
                sort!(T2, [:tick, :agent_id])
                @test T1 == T2
            end
        end
    end

    @testset "strict mode: out-of-range tick throws ReplayMissingError" begin
        cfg = load_world(REPLAY_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp, "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain_live = live_brain(cfg; dispatch = _replay_stub(0.01), max_concurrency = 8)
                run_simulation(cfg; brain = brain_live, run_id = "strict_src", n_steps = 2)

                rb = replay_brain("strict_src"; strict = true)
                @test_throws ReplayMissingError run_simulation(cfg; brain = rb, n_steps = 7)
            end
        end
    end

    @testset "non-strict mode: out-of-range ticks fall back to delta = 0" begin
        cfg = load_world(REPLAY_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp, "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain_live = live_brain(cfg; dispatch = _replay_stub(0.01), max_concurrency = 8)
                run_simulation(cfg; brain = brain_live, run_id = "ns_src", n_steps = 2)

                rb = replay_brain("ns_src"; strict = false)
                T = run_simulation(cfg; brain = rb, n_steps = 7)
                @test T isa DataFrame
                @test nrow(T) == cfg.n_agents * (7 + 1)
            end
        end
    end

    @testset "missing source: ArgumentError on nonexistent run_id" begin
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                @test_throws ArgumentError replay_brain("does-not-exist")
            end
        end
    end

    @testset "read-only: replay run writes no transcript rows" begin
        cfg = load_world(REPLAY_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp, "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain_live = live_brain(cfg; dispatch = _replay_stub(0.01), max_concurrency = 8)
                run_simulation(cfg; brain = brain_live, run_id = "ro_src", n_steps = 2)

                rb = replay_brain("ro_src")
                run_simulation(cfg; brain = rb, run_id = "ro_replay", n_steps = 2)

                open_store("ro_replay") do db
                    n = only(DBInterface.execute(
                        db, "SELECT COUNT(*) AS n FROM transcripts"
                    )).n
                    @test n == 0
                end
            end
        end
    end
end
