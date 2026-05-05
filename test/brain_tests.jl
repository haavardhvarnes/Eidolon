using Eidolon
using DataFrames
using DBInterface
using DuckDB
using Random: Xoshiro
using Test

const BRAIN_BASELINE_PATH = joinpath(@__DIR__, "..", "data", "seeds", "baseline.json")

# Generic helper: count rows matching a WHERE clause.
function _count_where(db, table, where)
    only(DBInterface.execute(db, "SELECT COUNT(*) AS n FROM $table $where")).n
end

# Pull all transcript rows as a DataFrame for inspection.
function _read_transcripts(db)
    return DataFrame(DBInterface.execute(
        db,
        """
        SELECT tick, agent_id, model, template, prompt, response,
               delta_raw, delta_clamped, status, latency_ms
        FROM transcripts
        ORDER BY tick, agent_id
        """
    ))
end

# Stub dispatch factory: returns a constant `delta` for every agent.
function _const_dispatch(delta::Real;
        memory::AbstractString = "stub-memory",
        reasoning::AbstractString = "stub-reasoning")
    d = Float64(delta)
    m = String(memory)
    r = String(reasoning)
    return (brain,
        aid,
        vars) -> (
        prompt = string("prompt-", aid),
        response = string("response-", aid),
        output = Eidolon.BrainOutput(d, m, r)
    )
end

@testset "brain" begin
    @testset "RandomBrain: same seed → identical trajectory" begin
        cfg = load_world(BRAIN_BASELINE_PATH)
        df1 = run_simulation(cfg; brain = RandomBrain(Xoshiro(7), 0.005))
        df2 = run_simulation(cfg; brain = RandomBrain(Xoshiro(7), 0.005))
        @test df1 == df2
        # And it actually moved opinions (vs. the NullBrain baseline).
        df_null = run_simulation(cfg)
        @test df_null != df1

        # Per-tick brain perturbations stay inside the Δ-cap.
        cap = Eidolon.brain_delta_cap()
        brain = RandomBrain(Xoshiro(11), 1.0)  # σ=1 ensures saturation
        model = initialize_world(cfg; brain = brain)
        for t in 1:5
            batch_reflect!(brain, model; tick = t)
            for a in Eidolon.allagents(model)
                @test abs(Eidolon.brain_perturbation(brain, a, model)) ≤ cap + 1.0e-12
            end
        end
    end

    @testset "LiveBrain happy path: transcripts populated and Δ reaches agents" begin
        cfg = load_world(BRAIN_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "0") do
                stub = _const_dispatch(0.01)
                brain = live_brain(cfg; dispatch = stub, max_concurrency = 8)
                df_live = run_simulation(cfg; brain = brain, run_id = "live")
                df_null = run_simulation(cfg; run_id = "null")

                @test brain.call_counter[] == cfg.n_agents * cfg.max_ticks
                @test df_live != df_null  # Δ reached agent.opinion

                open_store("live") do db
                    @test _count_where(
                        db, "transcripts",
                        "WHERE status = 'ok'"
                    ) == cfg.n_agents * cfg.max_ticks
                    @test _count_where(db, "transcripts", "") ==
                          cfg.n_agents * cfg.max_ticks
                end
            end
        end
    end

    @testset "Δ clamp: out-of-bound delta clamped, raw persisted" begin
        cfg = load_world(BRAIN_BASELINE_PATH)
        cap = Eidolon.brain_delta_cap()
        target_id = 1
        # Stub returns Δ=10.0 only for agent #1, Δ=0 for everyone else.
        stub = function (brain, aid, vars)
            δ = aid == target_id ? 10.0 : 0.0
            return (
                prompt = "p$aid",
                response = "r$aid",
                output = Eidolon.BrainOutput(δ, "", "")
            )
        end
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "0") do
                brain = live_brain(cfg; dispatch = stub, max_concurrency = 4)
                run_simulation(cfg; brain = brain, run_id = "clamp", n_steps = 2)

                open_store("clamp") do db
                    rows = DataFrame(DBInterface.execute(
                        db,
                        """
                        SELECT delta_raw, delta_clamped
                        FROM transcripts
                        WHERE agent_id = $target_id
                        """
                    ))
                    @test all(rows.delta_raw .== 10.0)
                    @test all(rows.delta_clamped .== cap)

                    # Other agents' rows: delta_raw == 0, clamped == 0.
                    others = DataFrame(DBInterface.execute(
                        db,
                        """
                        SELECT delta_raw, delta_clamped
                        FROM transcripts
                        WHERE agent_id <> $target_id
                        """
                    ))
                    @test all(others.delta_raw .== 0.0)
                    @test all(others.delta_clamped .== 0.0)
                end
            end
        end
    end

    @testset "cost cap: EIDOLON_MAX_LLM_CALLS triggers LLMBudgetExceeded" begin
        cfg = load_world(BRAIN_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "10") do
                stub = _const_dispatch(0.0)
                # Concurrency=1 keeps the failure deterministic on call #11.
                brain = live_brain(cfg; dispatch = stub, max_concurrency = 1)
                @test_throws LLMBudgetExceeded run_simulation(
                    cfg; brain = brain, run_id = "budget"
                )
            end
        end
    end

    @testset "schema drift: >5% raises, ≤5% falls back" begin
        cfg = load_world(BRAIN_BASELINE_PATH)

        # 6/100 agents fail parsing → drift_rate 0.06 > 0.05 → throw.
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "0") do
                fail_set_above = Set(1:6)
                stub_above = function (brain, aid, vars)
                    if aid in fail_set_above
                        throw(Eidolon.SchemaParseError("bad json", "{ broken"))
                    end
                    return (
                        prompt = "p$aid", response = "r$aid",
                        output = Eidolon.BrainOutput(0.0, "", "")
                    )
                end
                brain = live_brain(
                    cfg; dispatch = stub_above, max_concurrency = 4
                )
                @test_throws SchemaDriftError run_simulation(
                    cfg; brain = brain, run_id = "drift_above"
                )
            end
        end

        # 1/100 agents fail → drift_rate 0.01 ≤ 0.05 → run completes,
        # that agent's row has status="schema_error".
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "0") do
                target_id = 7
                stub_below = function (brain, aid, vars)
                    if aid == target_id
                        throw(Eidolon.SchemaParseError("bad json", "{ broken"))
                    end
                    return (
                        prompt = "p$aid", response = "r$aid",
                        output = Eidolon.BrainOutput(0.0, "", "")
                    )
                end
                brain = live_brain(
                    cfg; dispatch = stub_below, max_concurrency = 4
                )
                run_simulation(
                    cfg; brain = brain, run_id = "drift_below", n_steps = 1
                )
                open_store("drift_below") do db
                    @test _count_where(
                        db, "transcripts",
                        "WHERE status = 'schema_error' AND agent_id = $target_id"
                    ) == 1
                    # Everyone else parsed cleanly.
                    @test _count_where(
                        db, "transcripts",
                        "WHERE status = 'ok' AND agent_id <> $target_id"
                    ) == cfg.n_agents - 1
                end
            end
        end
    end

    @testset "transient retry: row marked retried, run completes" begin
        cfg = load_world(BRAIN_BASELINE_PATH)
        mktempdir() do tmp
            withenv("EIDOLON_RUNS_ROOT" => tmp,
                "EIDOLON_MAX_LLM_CALLS" => "0") do
                target_id = 13
                attempts = Ref(0)
                stub = function (brain, aid, vars)
                    if aid == target_id
                        attempts[] += 1
                        if attempts[] ≤ 2
                            throw(Eidolon.RetryableError("boom"))
                        end
                    end
                    return (
                        prompt = "p$aid", response = "r$aid",
                        output = Eidolon.BrainOutput(0.0, "", "")
                    )
                end
                # Use a tiny delta_cap so backoff sleeps are 0.5s × 1 = 0.5s
                # — keep the test bearable even with two retries.
                brain = live_brain(
                    cfg; dispatch = stub, max_concurrency = 4,
                    max_attempts = 3
                )
                run_simulation(
                    cfg; brain = brain, run_id = "retry", n_steps = 1
                )

                open_store("retry") do db
                    target_row = only(DBInterface.execute(
                        db,
                        """
                        SELECT status FROM transcripts
                        WHERE agent_id = $target_id AND tick = 1
                        """
                    ))
                    @test target_row.status == "retried"
                    # Other agents OK on first attempt.
                    @test _count_where(
                        db, "transcripts",
                        "WHERE status = 'ok' AND agent_id <> $target_id"
                    ) == cfg.n_agents - 1
                end
            end
        end
    end
end
