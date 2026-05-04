using Eidolon
using JSON
using Test

const SEEDS_DIR = joinpath(@__DIR__, "..", "data", "seeds")
const BASELINE_PATH = joinpath(SEEDS_DIR, "baseline.json")

# Round-trip helper: parse the baseline seed, let the caller mutate it,
# write to a temp file, then call `load_world`. Returns the result of
# the call (or rethrows the exception). Using a temp file keeps tests
# on the public API rather than reaching into hydration internals.
function _load_with_mutation(mutate!::Function)
    raw = JSON.parsefile(BASELINE_PATH)
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

@testset "load_world" begin
    @testset "happy path: baseline.json" begin
        cfg = load_world(BASELINE_PATH)
        @test cfg isa WorldConfig
        @test cfg.name == "baseline"
        @test cfg.seed == 42
        @test cfg.n_agents == 100
        @test cfg.max_ticks == 50
        @test length(cfg.personas) == 3
        @test Set(p.id for p in cfg.personas) == Set(["skeptic", "loyalist", "neutral"])
        @test cfg.topology.kind == "watts_strogatz"
        @test cfg.topology.params["k"] == 6
        @test cfg.topology.params["beta"] ≈ 0.10
        @test sum(values(cfg.persona_distribution)) ≈ 1.0
        @test length(cfg.interventions) == 1
        only_iv = cfg.interventions[1]
        @test only_iv.tick == 25
        @test only_iv.kind == "broadcast"
        @test only_iv.payload["intensity"] ≈ 0.5
        @test occursin("official statement", only_iv.payload["message"])
    end

    @testset "persona_distribution must sum to 1.0" begin
        err = try
            _load_with_mutation(raw -> raw["persona_distribution"]["loyalist"] = 0.10)
            nothing
        catch e
            e
        end
        @test err isa SchemaError
        @test occursin("sum to 1.0", err.msg)
    end

    @testset "persona_distribution keys must match a persona id" begin
        err = try
            _load_with_mutation() do raw
                # Replace a known key with an unknown one, preserving the sum.
                raw["persona_distribution"]["nonexistent"] = raw["persona_distribution"]["neutral"]
                delete!(raw["persona_distribution"], "neutral")
            end
            nothing
        catch e
            e
        end
        @test err isa SchemaError
        @test occursin("nonexistent", err.msg)
        @test occursin("persona_distribution", err.msg)
    end

    @testset "confidence_radius must be in (0, 1]" begin
        for bad_value in (0.0, 1.5)
            err = try
                _load_with_mutation(raw -> raw["personas"][1]["confidence_radius"] = bad_value)
                nothing
            catch e
                e
            end
            @test err isa SchemaError
            @test occursin("confidence_radius", err.msg)
        end
    end

    @testset "update_weight must be in [0, 1]" begin
        for bad_value in (-0.1, 1.5)
            err = try
                _load_with_mutation(raw -> raw["personas"][1]["update_weight"] = bad_value)
                nothing
            catch e
                e
            end
            @test err isa SchemaError
            @test occursin("update_weight", err.msg)
        end
    end

    @testset "watts_strogatz k must be even" begin
        err = try
            _load_with_mutation(raw -> raw["topology"]["params"]["k"] = 5)
            nothing
        catch e
            e
        end
        @test err isa SchemaError
        @test occursin("k to be even", err.msg)
    end

    @testset "watts_strogatz k must satisfy 0 ≤ k < n_agents" begin
        err = try
            # k is even and ≥ n_agents — triggers the bound check, not the parity check.
            _load_with_mutation(raw -> raw["topology"]["params"]["k"] = 100)
            nothing
        catch e
            e
        end
        @test err isa SchemaError
        @test occursin("0 ≤ k < n_agents", err.msg)
    end

    @testset "intervention tick must be in [0, max_ticks]" begin
        err = try
            _load_with_mutation(raw -> raw["interventions"][1]["tick"] = 9999)
            nothing
        catch e
            e
        end
        @test err isa SchemaError
        @test occursin("tick", err.msg)
        @test occursin("max_ticks", err.msg)
    end
end
