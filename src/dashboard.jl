# Phase 5 — Stipple.jl web dashboard (ADR-0003).
#
# Architecture: cooperative @async worker + reactive fields.
# NullBrain only — each tick is sub-millisecond so yield() between
# ticks keeps the WebSocket event loop responsive without needing a
# separate OS thread (ADR-0003 §Revised approach).

using Agents: abmspace, allagents, step!
using Graphs: edges, nv, src, dst
using Statistics: mean, var
using Stipple, Stipple.ReactiveTools, StipplePlotly, StippleUI

# --- Module-level simulation state ----------------------------------
# Scoped to one active run; reset by _launch_simulation! on each Start.

const _DASHBOARD_CFG  = Ref{Union{Nothing, WorldConfig}}(nothing)
const _RUNNING_CFG    = Ref{Union{Nothing, WorldConfig}}(nothing)
const _RUNNING_ABM    = Ref{Any}(nothing)
const _NODE_POSITIONS = Ref{Vector{Tuple{Float64, Float64}}}(Tuple{Float64, Float64}[])

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

# --- Graph layout helpers -------------------------------------------

# Circular layout: vertices evenly spaced on a unit circle.
function _circular_positions(n::Int)::Vector{Tuple{Float64, Float64}}
    [(cos(2π * (i - 1) / n), sin(2π * (i - 1) / n)) for i in 1:n]
end

# Build the static edge trace (gray lines, reused every tick).
function _edge_trace(g, positions::Vector{Tuple{Float64, Float64}})::PlotData
    xs = Float64[]
    ys = Float64[]
    for e in edges(g)
        push!(xs, positions[src(e)][1], positions[dst(e)][1], NaN)
        push!(ys, positions[src(e)][2], positions[dst(e)][2], NaN)
    end
    return PlotData(
        x     = xs,
        y     = ys,
        plot  = StipplePlotly.Charts.PLOT_TYPE_SCATTER,
        mode  = "lines",
        line  = PlotlyLine(color = "#cccccc", width = 0.8),
        hoverinfo = "none",
        name  = "edges",
        showlegend = false
    )
end

# Build the dynamic node trace (colored by opinion, rebuilt each tick).
function _node_trace(
        positions::Vector{Tuple{Float64, Float64}},
        agent_opinions::Vector{Float64}
)::PlotData
    n = length(positions)
    labels = ["Agent $i: $(round(agent_opinions[i]; digits=3))" for i in 1:n]
    return PlotData(
        x    = [p[1] for p in positions],
        y    = [p[2] for p in positions],
        plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER,
        mode = "markers",
        marker = PlotDataMarker(
            color      = agent_opinions,
            colorscale = "RdBu",
            cmin       = 0.0,
            cmax       = 1.0,
            size       = 10,
            showscale  = true
        ),
        text      = labels,
        hoverinfo = "text",
        name      = "agents"
    )
end

# PlotLayout for the opinion time-series chart.
const _CHART_LAYOUT = PlotLayout(
    title  = PlotLayoutTitle(text = "Mean Opinion Over Time"),
    xaxis  = [PlotLayoutAxis(title_text = "Tick")],
    yaxis  = [PlotLayoutAxis(title_text = "Opinion", range = [0.0, 1.0])]
)

# PlotLayout for the network graph (axes hidden).
const _NETWORK_LAYOUT = PlotLayout(
    title      = PlotLayoutTitle(text = "Opinion Network"),
    showlegend = false,
    xaxis      = [PlotLayoutAxis(showgrid = false, zeroline = false, showticklabels = false)],
    yaxis      = [PlotLayoutAxis(showgrid = false, zeroline = false, showticklabels = false)]
)

# --- Reactive model -------------------------------------------------

@app DashboardModel begin
    # Browser-writable inputs
    @in sim_status     = "idle"    # "idle" | "running" | "paused" | "done"
    @in do_pause       = false
    @in do_resume      = false
    @in do_reset       = false
    @in do_broadcast   = false
    @in broadcast_text = ""

    # Server-pushed display outputs
    @out current_tick      = 0
    @out max_ticks_display = 0
    @out mean_opinion      = 0.5
    @out opinion_variance  = 0.0
    @out event_log         = String[]

    # Chart data (time-series + network)
    @out opinion_chart   = [PlotData()]
    @out chart_layout    = _CHART_LAYOUT
    @out network_traces  = [PlotData(), PlotData()]
    @out network_layout  = _NETWORK_LAYOUT

    # Non-reactive private flags (never serialised to the browser)
    @private _paused   = false
    @private _stopping = false

    # --- Handlers ---

    @onchange sim_status begin
        if sim_status == "running"
            _paused   = false
            _stopping = false
            _launch_simulation!(__model__)
        end
    end

    @onchange do_pause begin
        if do_pause
            _paused    = true
            sim_status = "paused"
            do_pause   = false
        end
    end

    @onchange do_resume begin
        if do_resume
            _paused    = false
            sim_status = "running"
            do_resume  = false
        end
    end

    @onchange do_reset begin
        if do_reset
            _stopping          = true
            _paused            = false
            sim_status         = "idle"
            current_tick       = 0
            mean_opinion       = 0.5
            opinion_variance   = 0.0
            event_log          = String[]
            opinion_chart      = [PlotData()]
            network_traces     = [PlotData(), PlotData()]
            _RUNNING_ABM[]     = nothing
            _RUNNING_CFG[]     = nothing
            _NODE_POSITIONS[]  = Tuple{Float64, Float64}[]
            do_reset           = false
        end
    end

    @onchange do_broadcast begin
        if do_broadcast && !isempty(broadcast_text)
            _inject_broadcast!(__model__)
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
    opinion_chart = [PlotData()]

    # Pre-compute graph layout positions (fixed for the run's lifetime).
    g = abmspace(abm).graph
    positions = _circular_positions(nv(g))
    _NODE_POSITIONS[] = positions
    init_ops = [Float64(a.opinion) for a in allagents(abm)]
    network_traces = [_edge_trace(g, positions), _node_trace(positions, init_ops)]

    @async _run_loop!(__model__, abm, cfg)
end

@handler DashboardModel function _run_loop!(abm, cfg::WorldConfig)
    ticks      = Int[]
    means      = Float64[]
    g          = abmspace(abm).graph
    positions  = _NODE_POSITIONS[]

    for t in 1:(cfg.max_ticks)
        # Poll pause flag cooperatively — yields to let WebSocket messages
        # through between ticks (ADR-0003 §Revised approach).
        while _paused
            yield()
        end
        _stopping && break

        step!(abm, 1)
        snap = _build_snapshot(abm, t)

        push!(ticks, t)
        push!(means, snap.mean_opinion)

        current_tick     = snap.tick
        mean_opinion     = snap.mean_opinion
        opinion_variance = snap.opinion_variance

        # Rebuild the opinion time-series trace in place.
        opinion_chart = [PlotData(
            x    = ticks,
            y    = means,
            plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER,
            mode = "lines",
            name = "mean opinion"
        )]

        # Update node colours (edge trace is static, reuse from index 1).
        network_traces = [
            network_traces[1],
            _node_trace(positions, snap.node_opinions)
        ]

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
    iv = Intervention(
        tick + 1, "broadcast",
        Dict{String, Any}("message" => broadcast_text)
    )
    push!(cfg.interventions, iv)
    push!(event_log, "tick $(tick + 1): $broadcast_text")
    broadcast_text = ""
end

# --- UI layout ------------------------------------------------------

function _ui(model::DashboardModel)
    [
        row([cell(col = 12, [h2("Eidolon Simulator")])]),

        # Controls + status
        row([
            cell(col = 4, [
                btn("Start",
                    @click("sim_status = 'running'"),
                    color = "positive",
                    @iif("sim_status === 'idle'")
                ),
                btn("Pause",
                    @click("do_pause = true"),
                    color = "warning",
                    @iif("sim_status === 'running'")
                ),
                btn("Resume",
                    @click("do_resume = true"),
                    color = "info",
                    @iif("sim_status === 'paused'")
                ),
                btn("Reset",
                    @click("do_reset = true"),
                    color = "negative"
                ),
            ]),
            cell(col = 8, [
                p(["Status: ",           strong(@text(:sim_status))]),
                p(["Tick: ",             strong(@text(:current_tick)),
                   " / ",               strong(@text(:max_ticks_display))]),
                p(["Mean opinion: ",     strong(@text(:mean_opinion))]),
                p(["Opinion variance: ", strong(@text(:opinion_variance))]),
            ]),
        ]),

        # Charts
        row([
            cell(col = 6, [
                plot(:opinion_chart, layout = :chart_layout),
            ]),
            cell(col = 6, [
                plot(:network_traces, layout = :network_layout),
            ]),
        ]),

        # Broadcast
        row([
            cell(col = 12, [
                h3("Broadcast"),
                textfield("Message", :broadcast_text),
                btn("Broadcast",
                    @click("do_broadcast = true"),
                    color = "primary",
                    @iif("sim_status !== 'idle'")
                ),
            ]),
        ]),

        # Event log
        row([
            cell(col = 12, [
                h3("Event Log"),
                Stipple.Html.ul([
                    Stipple.Html.li(@recur("entry in event_log"), @text("entry"))
                ]),
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

"""
    stop_dashboard()

Shut down the Genie/Stipple server started by [`start_dashboard`](@ref).
"""
function stop_dashboard()
    Stipple.down()
    return nothing
end
