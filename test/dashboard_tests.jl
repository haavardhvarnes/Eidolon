using Agents: allagents, step!
using Eidolon
using Test

# Minimal WorldConfig for dashboard tests.
function _dash_cfg(; n_agents = 10, max_ticks = 4, seed = 42)
    persona = AgentPersona("neutral", "test", 0.5, 0.1, 0.4, 0.35, 8, String[])
    topo    = GraphTopology("watts_strogatz", Dict{String, Any}("k" => 4, "beta" => 0.1))
    WorldConfig(
        "dash-test", "dashboard test fixture", seed, n_agents, max_ticks,
        [persona], Dict{String, Float64}("neutral" => 1.0),
        topo, "claude-haiku-4-5-20251001", Intervention[]
    )
end

@testset "dashboard" begin
    @testset "TickSnapshot construction" begin
        snap = TickSnapshot(3, 0.6, 0.04, [0.4, 0.6, 0.8])
        @test snap.tick == 3
        @test snap.mean_opinion ≈ 0.6
        @test snap.opinion_variance ≈ 0.04
        @test snap.node_opinions == [0.4, 0.6, 0.8]
    end

    @testset "_build_snapshot: mean and variance match allagents" begin
        cfg = _dash_cfg()
        abm = initialize_world(cfg)
        snap = Eidolon._build_snapshot(abm, 0)
        @test snap.tick == 0
        ops = [a.opinion for a in allagents(abm)]
        @test snap.mean_opinion ≈ sum(ops) / length(ops)
        @test snap.opinion_variance ≥ 0.0
        @test length(snap.node_opinions) == cfg.n_agents
    end

    @testset "_fresh_cfg: copies interventions, preserves fields" begin
        iv  = Intervention(1, "broadcast", Dict{String, Any}("message" => "hi"))
        cfg = WorldConfig(
            "x", "y", 1, 5, 10,
            [AgentPersona("a", "b", 0.5, 0.1, 0.4, 0.3, 8, String[])],
            Dict{String, Float64}("a" => 1.0),
            GraphTopology("watts_strogatz", Dict{String, Any}("k" => 4, "beta" => 0.1)),
            "m", [iv]
        )
        fresh = Eidolon._fresh_cfg(cfg)
        # Same field values
        @test fresh.name    == cfg.name
        @test fresh.seed    == cfg.seed
        @test fresh.n_agents == cfg.n_agents
        # Different vector object — mutations don't bleed between runs
        @test fresh.interventions !== cfg.interventions
        @test length(fresh.interventions) == 1
        push!(fresh.interventions, Intervention(2, "broadcast", Dict{String, Any}()))
        @test length(cfg.interventions) == 1     # original untouched
        @test length(fresh.interventions) == 2
    end

    @testset "simulation produces correct tick count without server" begin
        # Verify the simulation loop logic (ABM + snapshot) works headlessly.
        cfg  = _dash_cfg(; n_agents = 8, max_ticks = 5)
        abm  = initialize_world(cfg)
        snaps = TickSnapshot[]
        for t in 1:(cfg.max_ticks)
            step!(abm, 1)
            push!(snaps, Eidolon._build_snapshot(abm, t))
        end
        @test length(snaps) == cfg.max_ticks
        @test snaps[end].tick == cfg.max_ticks
        @test all(s -> 0.0 ≤ s.mean_opinion ≤ 1.0, snaps)
        @test all(s -> s.opinion_variance ≥ 0.0, snaps)
        @test all(s -> length(s.node_opinions) == cfg.n_agents, snaps)
    end

    @testset "broadcast injection appends to running cfg" begin
        cfg   = _dash_cfg()
        fresh = Eidolon._fresh_cfg(cfg)
        iv    = Intervention(3, "broadcast", Dict{String, Any}("message" => "hello"))
        push!(fresh.interventions, iv)
        @test length(fresh.interventions) == 1
        @test fresh.interventions[1].kind == "broadcast"
        @test fresh.interventions[1].payload["message"] == "hello"
        # Original cfg unchanged
        @test isempty(cfg.interventions)
    end
end
