using Eidolon
using DataFrames
using Statistics: mean, var
using Test

# Minimal WorldConfig for sensitivity tests: 20 agents, 5 ticks, two
# personas so we can exercise multi-persona ranking.
function _sens_cfg()
    hawk = AgentPersona("hawk", "high-confidence extremist",
        0.8, 0.05, 0.15, 0.6, 8, String[])
    dove = AgentPersona("dove", "open-minded moderate",
        0.5, 0.1, 0.45, 0.3, 8, String[])
    topo = GraphTopology("watts_strogatz", Dict{String, Any}("k" => 4, "beta" => 0.1))
    return WorldConfig(
        "sens-test", "sensitivity fixture", 7, 20, 5,
        [hawk, dove],
        Dict{String, Float64}("hawk" => 0.4, "dove" => 0.6),
        topo,
        "claude-haiku-4-5-20251001",
        Intervention[]
    )
end

@testset "sensitivity" begin
    @testset "opinion_variance_metric: uses final tick only" begin
        df = DataFrame(
            tick       = [0, 0, 1, 1],
            agent_id   = [1, 2, 1, 2],
            opinion    = [0.2, 0.8, 0.3, 0.7],
            persona_id = ["dove", "dove", "dove", "dove"]
        )
        v = opinion_variance_metric(df)
        @test v ≈ var([0.3, 0.7]; corrected = false)
        # tick-0 opinions must not influence the result
        @test v != var([0.2, 0.8]; corrected = false)
    end

    @testset "SensitivityDimension construction" begin
        d = SensitivityDimension("hawk", :confidence_radius, 0.1, 0.5)
        @test d.persona_id == "hawk"
        @test d.param == :confidence_radius
        @test d.lo == 0.1
        @test d.hi == 0.5
    end

    @testset "persona_sensitivity: returns well-formed SensitivityResult" begin
        cfg = _sens_cfg()
        dims = [
            SensitivityDimension("hawk", :confidence_radius, 0.05, 0.4),
            SensitivityDimension("dove", :confidence_radius, 0.1, 0.6),
        ]
        # n_samples=16 keeps total evals at 16*(2+2)=64 — fast in CI.
        result = persona_sensitivity(cfg, dims; n_samples = 16, n_steps = 3)

        @test result isa SensitivityResult
        @test result.metric_name == "opinion_variance"
        @test length(result.S1) == 2
        @test length(result.ST) == 2
        @test length(result.dimensions) == 2
        @test all(isfinite, result.S1)
        @test all(isfinite, result.ST)
    end

    @testset "persona_sensitivity: custom metric" begin
        cfg = _sens_cfg()
        dims = [SensitivityDimension("hawk", :update_weight, 0.1, 0.8)]
        mean_metric(df) = begin
            final_tick = maximum(df.tick)
            mean(filter(r -> r.tick == final_tick, df).opinion)
        end
        result = persona_sensitivity(
            cfg, dims;
            metric       = mean_metric,
            metric_name  = "mean_opinion",
            n_samples    = 16,
            n_steps      = 2
        )
        @test result.metric_name == "mean_opinion"
        @test length(result.S1) == 1
        @test isfinite(result.ST[1])
    end

    @testset "persona_sensitivity: validation errors" begin
        cfg = _sens_cfg()
        # Empty dims
        @test_throws ArgumentError persona_sensitivity(cfg, SensitivityDimension[])
        # Unknown persona
        @test_throws ArgumentError persona_sensitivity(
            cfg,
            [SensitivityDimension("nobody", :confidence_radius, 0.1, 0.5)]
        )
        # Unsupported param
        @test_throws ArgumentError persona_sensitivity(
            cfg,
            [SensitivityDimension("hawk", :memory_capacity, 1.0, 10.0)]
        )
        # lo ≥ hi
        @test_throws ArgumentError persona_sensitivity(
            cfg,
            [SensitivityDimension("hawk", :confidence_radius, 0.5, 0.1)]
        )
        @test_throws ArgumentError persona_sensitivity(
            cfg,
            [SensitivityDimension("hawk", :confidence_radius, 0.3, 0.3)]
        )
    end

    @testset "top_personas: ranking and aggregation" begin
        dims = [
            SensitivityDimension("hawk",    :confidence_radius, 0.1, 0.4),
            SensitivityDimension("dove",    :confidence_radius, 0.1, 0.6),
            SensitivityDimension("dove",    :update_weight,     0.1, 0.8),
            SensitivityDimension("neutral", :update_weight,     0.1, 0.7),
        ]
        # ST manually set: hawk=0.2, dove=0.35+0.25=0.60, neutral=0.10
        result = SensitivityResult(
            dims,
            [0.1, 0.3, 0.2, 0.05],
            [0.2, 0.35, 0.25, 0.10],
            "test_metric"
        )
        top = top_personas(result; n = 3)

        @test length(top) == 3
        @test top[1].persona_id == "dove"
        @test top[1].total_sensitivity ≈ 0.60
        @test top[2].persona_id == "hawk"
        @test top[2].total_sensitivity ≈ 0.20
        @test top[3].persona_id == "neutral"
        @test top[3].total_sensitivity ≈ 0.10
    end

    @testset "top_personas: n larger than persona count" begin
        dims = [SensitivityDimension("solo", :confidence_radius, 0.1, 0.5)]
        result = SensitivityResult(dims, [0.9], [0.95], "test")
        top = top_personas(result; n = 5)
        @test length(top) == 1
        @test top[1].persona_id == "solo"
    end

    @testset "top_personas: single-dimension result" begin
        cfg = _sens_cfg()
        dims = [SensitivityDimension("hawk", :confidence_radius, 0.05, 0.4)]
        result = persona_sensitivity(cfg, dims; n_samples = 16, n_steps = 3)
        top = top_personas(result; n = 3)
        @test length(top) == 1
        @test top[1].persona_id == "hawk"
        @test isfinite(top[1].total_sensitivity)
    end
end
