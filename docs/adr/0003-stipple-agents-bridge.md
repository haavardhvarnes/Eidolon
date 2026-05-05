# ADR-0003: Stipple ↔ Agents.jl event-loop bridge

## Status
Accepted — 2026-05-05

## Context

Phase 5 adds a Stipple.jl (GenieFramework) web dashboard to the project.
Two event loops need to co-exist without blocking each other:

- **Agents.jl loop** — `step!(model, 1)` is a synchronous, blocking call.
  A full run of N ticks takes seconds to minutes depending on brain type.
- **Stipple / Genie loop** — the HTTP + WebSocket server that keeps the
  browser reactive. If this thread stalls, the UI freezes.

Secondary constraints:

- **Pause / resume** must interrupt the simulation mid-run, not just abort it.
- **Broadcast injection** must be safe to call from the UI while the
  simulation is running (no data race on `model.cfg.interventions`).
- **NullBrain is the only supported brain in the dashboard** — live LLM
  calls during an interactive session are deferred; the UI is a demo of
  the dynamics, not a production inference system.
- No new concurrency libraries: the standard library's `Threads`,
  `Channel`, and `Base.Atomic` are sufficient.

### Alternatives considered

**A. Timer-driven single-step on Stipple's event handler.**
Register an `@onchange` or Stipple timer that calls `step!(model, 1)`
directly. Simple, but each tick call blocks the WebSocket handler for its
duration. Fine for microsecond ticks, unacceptable once `batch_reflect!`
is in the loop (seconds per tick for large runs).

**B. `@async` coroutine.**
Launch an `@async` task from the handler. Julia's cooperative scheduler
will yield back to the HTTP event loop only at explicit yield points
(`sleep`, channel operations). Agents.jl's step loop has no yield points,
so this degenerates to option A under load.

**C. `Threads.@spawn` worker + `Channel` snapshot queue (chosen).**
The simulation runs in a dedicated OS thread. Tick snapshots are sent
through a `Channel{TickSnapshot}` to a Stipple timer callback (every
~500 ms) that drains the channel and updates the reactive model.
Pause/resume is an `Atomic{Bool}` flag polled at the top of each tick
iteration; reset closes and replaces the channel.

**D. Separate Julia process via `Distributed`.**
Overkill for a single-machine demo. Adds serialization cost and process
management complexity with no benefit over option C for v1.

## Decision

Use **option C**: a `Threads.@spawn` simulation worker + `Channel`
snapshot queue.

### Bridge contract

```julia
# In the Stipple reactive model:
mutable struct DashboardModel <: ReactiveModel
    # --- UI-facing reactive fields ---
    sim_status::R{String}       # "idle" | "running" | "paused" | "done"
    current_tick::R{Int}
    mean_opinion::R{Float64}
    opinion_variance::R{Float64}
    broadcast_text::R{String}

    # --- Internal (non-reactive) ---
    _worker::Union{Nothing, Task}
    _abm::Union{Nothing, StandardABM}
    _pause_flag::Threads.Atomic{Bool}
    _reset_flag::Threads.Atomic{Bool}
    _snapshots::Channel{TickSnapshot}
end
```

`TickSnapshot` is a plain struct (tick, mean_opinion, opinion_variance,
node_opinions) passed by value through the channel — no shared mutable
state between the worker thread and the Stipple handler thread.

### Worker lifecycle

```
Start  → spawn worker task; set sim_status = "running"
Pause  → set _pause_flag = true; worker polls flag at top of each tick
Resume → set _pause_flag = false
Reset  → set _reset_flag = true; wait for worker to exit; rebuild ABM
Stop   → set _reset_flag = true (same path as reset, but stay idle)
```

The worker loops:

```julia
for t in 1:cfg.max_ticks
    while model._pause_flag[]
        sleep(0.05)          # yield; do not burn CPU
    end
    model._reset_flag[] && break
    batch_reflect!(brain, abm; tick = t)
    step!(abm, 1)
    snapshot = TickSnapshot(t, _mean_opinion(abm), _opinion_variance(abm),
                            _node_opinions(abm))
    put!(model._snapshots, snapshot)   # non-blocking (buffered channel)
end
```

### Stipple timer (UI refresh)

A Stipple `@periodic 500` (or equivalent) callback on the server side
drains `_snapshots` and pushes the latest values to the reactive fields.
This is the only place reactive fields are written from — safe because
Stipple's internal lock protects reactive field writes.

### Broadcast injection

The broadcast button handler appends an `Intervention` to the **running
ABM's** `cfg.interventions` vector (the running model holds a mutable
copy). The worker reads `_global_event(model, tick)` at the start of each
tick's `batch_reflect!` call, so injected broadcasts take effect on the
next tick with no extra synchronisation needed.

## Consequences

**Upsides**

- Stipple's event loop never blocks — the worker runs on a separate OS
  thread, yielding only at `sleep(0.05)` during pause.
- Pause / resume is cheap: one `Atomic{Bool}` read per tick.
- The `Channel` acts as a natural buffer; the UI can update at 2 Hz
  while the simulation runs at whatever rate the hardware allows.
- `TickSnapshot` is copied by value — no lock needed between the worker
  and the Stipple timer callback.

**Downsides / follow-ups**

- `cfg.interventions` mutation during a live run is not thread-safe if
  multiple broadcasts fire at exactly the same tick boundary. Acceptable
  for v1 (single-user demo); add a `ReentrantLock` if this becomes an
  issue.
- `Threads.@spawn` requires Julia ≥ 1.3 (already satisfied by
  `julia = "1.10"` in `Project.toml`).
- The `@periodic` Stipple API may differ across Stipple minor versions —
  pin the compat bound carefully and test on the version in `Manifest.toml`.
- Dashboard performance has not been profiled for large agents (> 500).
  The StipplePlotly graph layout is O(n) in nodes; for very large runs
  consider down-sampling to the top-k opinion clusters.
