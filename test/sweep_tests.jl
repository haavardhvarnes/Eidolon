using Eidolon
using DataFrames
using Test

# Run f() under a SimpleLogger capturing Warn+ messages.
# Returns (result_of_f, captured_log_string).
function _capture_warnings(f::Function)
    buf = IOBuffer()
    logger = Base.CoreLogging.SimpleLogger(buf, Base.CoreLogging.Warn)
    result = Base.CoreLogging.with_logger(f, logger)
    return result, String(take!(buf))
end

# Minimal WorldConfig for fast sweep tests (10 agents, 2 ticks).
function _sweep_tiny_cfg(;
        name::AbstractString = "tiny",
        n_agents::Int = 10,
        max_ticks::Int = 2,
        seed::Int = 99
)
    persona = AgentPersona(
        "neutral", "test persona", 0.5, 0.1, 0.4, 0.35, 8, String[])
    topo = GraphTopology(
        "watts_strogatz", Dict{String, Any}("k" => 4, "beta" => 0.1))
    return WorldConfig(
        name,
        "sweep test fixture",
        seed,
        n_agents,
        max_ticks,
        [persona],
        Dict{String, Float64}("neutral" => 1.0),
        topo,
        "claude-haiku-4-5-20251001",
        Intervention[]
    )
end

# Stub dispatch: constant delta for every agent.
function _sweep_stub(delta::Real)
    d = Float64(delta)
    return (brain, aid,
        vars) -> (
        prompt = "p$aid",
        response = "r$aid",
        output = BrainOutput(d, "", "")
    )
end

@testset "sweep" begin
    @testset "auto_run_id: format, hash provenance, name uniqueness" begin
        cfg = _sweep_tiny_cfg()
        id1 = auto_run_id(cfg)
        sleep(1)  # ensure the timestamp segment differs
        id2 = auto_run_id(cfg)

        # Format: <name>-<8digits>-<6digits>-<8hexchars>
        pat = r"^.+-\d{8}-\d{6}-[0-9a-f]{8}$"
        @test occursin(pat, id1)
        @test occursin(pat, id2)

        # Same cfg → same hash component (last dash-segment)
        hash1 = split(id1, "-")[end]
        hash2 = split(id2, "-")[end]
        @test hash1 == hash2

        # After sleep(1) the timestamp segment differs → ids differ
        @test id1 != id2

        # Different name → different cfg JSON → different hash
        cfg2 = _sweep_tiny_cfg(; name = "other")
        id3 = auto_run_id(cfg2)
        @test split(id3, "-")[end] != hash1
    end

    @testset "expand_grid: Cartesian product and lexicographic order" begin
        spec = (sigma = [0.01, 0.05], confidence_radius = [0.2, 0.4])
        grid = expand_grid(spec)

        @test length(grid) == 4
        @test grid[1] == (sigma = 0.01, confidence_radius = 0.2)
        @test grid[2] == (sigma = 0.01, confidence_radius = 0.4)
        @test grid[3] == (sigma = 0.05, confidence_radius = 0.2)
        @test grid[4] == (sigma = 0.05, confidence_radius = 0.4)

        # Empty value vector on any field → empty result
        @test isempty(expand_grid((a = Int[], b = [1, 2])))
        @test isempty(expand_grid((a = [1, 2], b = Int[])))
    end

    @testset "load_trajectories: happy path — two runs" begin
        cfg = _sweep_tiny_cfg()
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                id1, id2 = "traj-run-a", "traj-run-b"
                run_simulation(cfg; run_id = id1)
                run_simulation(cfg; run_id = id2)

                df = load_trajectories([id1, id2])
                expected = 2 * cfg.n_agents * (cfg.max_ticks + 1)
                @test nrow(df) == expected
                @test Set(df.run_id) == Set([id1, id2])
                @test names(df) == ["run_id", "tick", "agent_id", "opinion", "persona_id"]
            end
        end
    end

    @testset "load_trajectories: missing id warns and is skipped" begin
        cfg = _sweep_tiny_cfg()
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                valid_id = "traj-valid"
                run_simulation(cfg; run_id = valid_id)

                df, log_out = _capture_warnings() do
                    load_trajectories([valid_id, "nonexistent-traj-run"])
                end
                @test occursin("nonexistent-traj-run", log_out)
                @test nrow(df) == cfg.n_agents * (cfg.max_ticks + 1)
                @test all(==(valid_id), df.run_id)
            end
        end
    end

    @testset "load_trajectories: empty input → empty DataFrame" begin
        df = load_trajectories(String[])
        @test df isa DataFrame
        @test nrow(df) == 0
        @test names(df) == ["run_id", "tick", "agent_id", "opinion", "persona_id"]
    end

    @testset "load_transcripts: happy path (stub LiveBrain)" begin
        cfg = _sweep_tiny_cfg()
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp, "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain = live_brain(cfg; dispatch = _sweep_stub(0.0), max_concurrency = 4)
                run_id = "transcript-run-a"
                run_simulation(cfg; brain = brain, run_id = run_id)

                df = load_transcripts([run_id])
                @test nrow(df) == cfg.n_agents * cfg.max_ticks
                @test all(==(run_id), df.run_id)
                expected_cols = [
                    "run_id", "tick", "agent_id", "model", "template",
                    "prompt", "response", "delta_raw", "delta_clamped",
                    "status", "latency_ms"
                ]
                @test names(df) == expected_cols
            end
        end
    end

    @testset "grid_sweep: 2-point grid, correct per-run row counts" begin
        base_cfg = _sweep_tiny_cfg()
        n_steps = 2

        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp) do
                grid = expand_grid((n_agents = [8, 12],))
                pairs = grid_sweep(
                    point -> WorldConfig(
                        "tiny",
                        "test",
                        99,
                        point.n_agents,
                        n_steps,
                        base_cfg.personas,
                        base_cfg.persona_distribution,
                        base_cfg.topology,
                        base_cfg.llm_model,
                        Intervention[]
                    ),
                    grid;
                    n_steps_factory = _ -> n_steps
                )

                @test length(pairs) == 2
                run_ids = [p[2] for p in pairs]
                @test length(unique(run_ids)) == 2

                df = load_trajectories(run_ids)

                point1, id1 = pairs[1]
                @test point1.n_agents == 8
                @test nrow(filter(r -> r.run_id == id1, df)) ==
                      8 * (n_steps + 1)

                point2, id2 = pairs[2]
                @test point2.n_agents == 12
                @test nrow(filter(r -> r.run_id == id2, df)) ==
                      12 * (n_steps + 1)
            end
        end
    end
end
