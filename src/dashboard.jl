# Phase 5 — Stipple.jl web dashboard (ADR-0003).
#
# Architecture: cooperative @async worker + reactive fields.
# NullBrain only — each tick is sub-millisecond so yield() between
# ticks keeps the WebSocket event loop responsive without needing a
# separate OS thread (ADR-0003 §Revised approach).

using Statistics: mean, var
using Stipple, Stipple.ReactiveTools, StipplePlotly, StippleUI

# --- Module-level simulation state ----------------------------------
# Scoped to one active run; reset by _launch_simulation! on each Start.

const _DASHBOARD_CFG = Ref{Union{Nothing, WorldConfig}}(nothing)
const _RUNNING_CFG   = Ref{Union{Nothing, WorldConfig}}(nothing)
const _RUNNING_ABM   = Ref{Any}(nothing)

# --- TickSnapshot ---------------------------------------------------

"""
    TickSnapshot(tick, mean_opinion, opinion_variance, node_opinions)

Plain value type produced by the simulation worker each tick and
consumed by the reactive-field update path (ADR-0003 §Bridge contract).
"""
struct TickSnapshot
    tick::Int
    mean_opinion::Float64
    opinion_variance::Float64
    node_opinions::Vector{Float64}
end

function _build_snapshot(abm, tick::Int)::TickSnapshot
    ops = [Float64(a.opinion) for a in allagents(abm)]
    return TickSnapshot(tick, mean(ops), var(ops; corrected = false), ops)
end

# Returns a fresh WorldConfig with a clean interventions copy so each
# run starts without broadcast residue from a previous session.
function _fresh_cfg(cfg::WorldConfig)::WorldConfig
    return WorldConfig(
        cfg.name, cfg.description, cfg.seed, cfg.n_agents, cfg.max_ticks,
        cfg.personas, cfg.persona_distribution, cfg.topology,
        cfg.llm_model, copy(cfg.interventions)
    )
end

# --- Reactive model -------------------------------------------------

@app DashboardModel begin
    # Browser-writable inputs
    @in sim_status    = "idle"    # "idle" | "running" | "paused" | "done"
    @in do_pause      = false
    @in do_resume     = false
    @in do_reset      = false
    @in do_broadcast  = false
    @in broadcast_text = ""

    # Server-pushed display outputs
    @out current_tick      = 0
    @out max_ticks_display = 0
    @out mean_opinion      = 0.5
    @out opinion_variance  = 0.0
    @out event_log         = String[]

    # Non-reactive private flags (never serialised to the browser)
    @private _paused   = false
    @private _stopping = false

    # --- Handlers ---

    @onchange sim_status begin
        if sim_status == "running"
            _paused   = false
            _stopping = false
            _launch_simulation!()
        end
    end

    @onchange do_pause begin
        if do_pause
            _paused   = true
            sim_status = "paused"
            do_pause  = false
        end
    end

    @onchange do_resume begin
        if do_resume
            _paused   = false
            sim_status = "running"
            do_resume = false
        end
    end

    @onchange do_reset begin
        if do_reset
            _stopping      = true
            _paused        = false
            sim_status     = "idle"
            current_tick   = 0
            mean_opinion   = 0.5
            opinion_variance = 0.0
            event_log      = String[]
            _RUNNING_ABM[] = nothing
            _RUNNING_CFG[] = nothing
            do_reset       = false
        end
    end

    @onchange do_broadcast begin
        if do_broadcast && !isempty(broadcast_text)
            _inject_broadcast!()
            do_broadcast = false
        end
    end
end

# --- Handler functions ----------------------------------------------

@handler DashboardModel function _launch_simulation!()
    base_cfg = _DASHBOARD_CFG[]
    if base_cfg === nothing
        sim_status = "idle"
        return
    end
    cfg = _fresh_cfg(base_cfg)
    _RUNNING_CFG[] = cfg
    abm = initialize_world(cfg)
    _RUNNING_ABM[] = abm
    max_ticks_display = cfg.max_ticks
    @async _run_loop!(abm, cfg)
end

@handler DashboardModel function _run_loop!(abm, cfg::WorldConfig)
    for t in 1:(cfg.max_ticks)
        # Poll pause flag cooperatively — yields to let WebSocket messages
        # through between ticks (ADR-0003 §Revised approach).
        while _paused
            yield()
        end
        _stopping && break
        step!(abm, 1)
        snap         = _build_snapshot(abm, t)
        current_tick      = snap.tick
        mean_opinion      = snap.mean_opinion
        opinion_variance  = snap.opinion_variance
        yield()
    end
    if !_stopping
        sim_status = "done"
    end
end

@handler DashboardModel function _inject_broadcast!()
    cfg = _RUNNING_CFG[]
    cfg === nothing && return
    tick = current_tick
    iv = Intervention(tick + 1, "broadcast",
                      Dict{String, Any}("message" => broadcast_text))
    push!(cfg.interventions, iv)
    push!(event_log, "tick $(tick + 1): $broadcast_text")
    broadcast_text = ""
end

# --- UI layout ------------------------------------------------------

function _ui(model::DashboardModel)
    [
        row([
            cell(col = 12, [
                h2("Eidolon Simulator"),
            ]),
        ]),
        row([
            cell(col = 4, [
                btn("Start",
                    @click("sim_status = 'running'"),
                    color = "positive",
                    [Symbol(":disable") => "sim_status !== 'idle'"]
                ),
                btn("Pause",
                    @click("do_pause = true"),
                    color = "warning",
                    [Symbol(":disable") => "sim_status !== 'running'"]
                ),
                btn("Resume",
                    @click("do_resume = true"),
                    color = "info",
                    [Symbol(":disable") => "sim_status !== 'paused'"]
                ),
                btn("Reset",
                    @click("do_reset = true"),
                    color = "negative"
                ),
            ]),
            cell(col = 8, [
                p(["Status: ", strong(@text(:sim_status))]),
                p(["Tick: ",   strong(@text(:current_tick)),
                   " / ",      strong(@text(:max_ticks_display))]),
                p(["Mean opinion: ",     strong(@text(:mean_opinion))]),
                p(["Opinion variance: ", strong(@text(:opinion_variance))]),
            ]),
        ]),
        row([
            cell(col = 12, [
                h3("Broadcast"),
                textfield("Message", :broadcast_text),
                btn("Broadcast",
                    @click("do_broadcast = true"),
                    color = "primary",
                    [Symbol(":disable") => "sim_status === 'idle'"]
                ),
            ]),
        ]),
    ]
end

# --- Public API -----------------------------------------------------

"""
    start_dashboard(cfg::WorldConfig; port::Int = 8080)

Launch the Stipple web dashboard for `cfg`. Opens a non-blocking Genie
server at `http://localhost:<port>`. Call [`Stipple.down()`](@ref) to
stop it.
"""
function start_dashboard(cfg::WorldConfig; port::Int = 8080)
    _DASHBOARD_CFG[] = cfg

    Stipple.route("/") do
        model = init(DashboardModel)
        page(model, _ui(model))
    end

    Stipple.up(port; async = true)
    @info "Eidolon dashboard → http://localhost:$port"
    return nothing
end
