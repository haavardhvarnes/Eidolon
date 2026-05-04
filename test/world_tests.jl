using Eidolon
using Agents: abmspace, allagents, nagents
using DataFrames
using Graphs: ne, nv
using JSON
using Statistics: var
using Test

const WORLD_SEEDS_DIR = joinpath(@__DIR__, "..", "data", "seeds")
const WORLD_BASELINE_PATH = joinpath(WORLD_SEEDS_DIR, "baseline.json")

# Same round-trip helper used in `io_tests.jl`: parse the baseline,
# mutate the raw dict, write to a tempfile, then call `load_world`.
function _world_load_with_mutation(mutate!::Function)
    raw = JSON.parsefile(WORLD_BASELINE_PATH)
    mutate!(raw)
    path, io = mktemp()
    try
        JSON.print(io, raw)
        close(io)
        return load_world(path)
    finally
        rm(path; force = true)
    end
end

@testset "world" begin
    @testset "initialize_world: shape and persona sampling" begin
        cfg = load_world(WORLD_BASELINE_PATH)
        model = initialize_world(cfg)

        @test nagents(model) == 100

        graph = abmspace(model).graph
        @test nv(graph) == cfg.n_agents
        # Watts–Strogatz preserves edge count = N·k/2 for any β ∈ [0, 1].
        @test ne(graph) == cfg.n_agents * Int(cfg.topology.params["k"]) ÷ 2

        persona_ids = Set(p.id for p in cfg.personas)
        for a in allagents(model)
            @test 0.0 ≤ a.opinion ≤ 1.0
            @test a.persona_id in persona_ids
        end
    end

    @testset "run_simulation: trajectory shape and bounds" begin
        cfg = load_world(WORLD_BASELINE_PATH)
        df = run_simulation(cfg)

        @test df isa DataFrame
        @test names(df) == ["tick", "agent_id", "opinion", "persona_id"]
        @test nrow(df) == cfg.n_agents * (cfg.max_ticks + 1)
        @test all(0.0 .≤ df.opinion .≤ 1.0)
        @test minimum(df.tick) == 0
        @test maximum(df.tick) == cfg.max_ticks
    end

    @testset "determinism under NullBrain" begin
        cfg = load_world(WORLD_BASELINE_PATH)
        df1 = run_simulation(cfg)
        df2 = run_simulation(cfg)
        @test df1 == df2
    end

    @testset "HK convergence sanity (single persona, broad ε)" begin
        cfg = _world_load_with_mutation() do raw
            raw["personas"] = [
                Dict{String, Any}(
                "id" => "neutral",
                "description" => "Single-persona convergence harness.",
                "opinion_prior_mean" => 0.5,
                "opinion_prior_std" => 0.2,
                "confidence_radius" => 0.5,
                "update_weight" => 0.35,
                "memory_capacity" => 0,
                "tags" => String[]
            ),
            ]
            raw["persona_distribution"] = Dict{String, Any}("neutral" => 1.0)
            raw["max_ticks"] = 200
            raw["interventions"] = Any[]
        end
        df = run_simulation(cfg)
        initial = df[df.tick .== 0, :opinion]
        final = df[df.tick .== cfg.max_ticks, :opinion]
        @test var(final) < var(initial)
    end
end
