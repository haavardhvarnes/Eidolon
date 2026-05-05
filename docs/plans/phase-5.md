# Phase 5 Plan: Web Control Panel

**Goal:** A Stipple.jl dashboard that lets a non-Julia user launch a
simulation, watch it run tick-by-tick, and inject broadcast interventions
from the browser — without touching the REPL.

**Key architectural decision:** ADR-0003 (`docs/adr/0003-stipple-agents-bridge.md`)
— `Threads.@spawn` worker + `Channel` snapshot queue. Read it before
starting any task below.

---

## Tasks (ordered)

### T1 — Dependencies
- Add `Stipple`, `StipplePlotly`, `StippleUI` to `Project.toml` and
  resolve `Manifest.toml`.
- Set compat bounds: `Stipple = "0"` (pre-1.0 ecosystem; pin to resolved
  minor in Manifest), same for `StipplePlotly`.
- Update `docs/adr/README.md` to mark ADR-0003 Accepted.

**Done when:** `julia --project=. -e 'using Stipple'` succeeds and CI still passes.

---

### T2 — `dashboard.jl` skeleton
- Define `TickSnapshot` struct in `src/dashboard.jl`.
- Define `DashboardModel <: ReactiveModel` per ADR-0003 §Bridge contract,
  with all reactive fields at their idle defaults.
- Register Stipple routes (`/` → dashboard page) and boot the server via
  `start_dashboard(cfg::WorldConfig; port = 8080)`.
- No simulation logic yet — just a page that loads in a browser.

**Done when:** `julia --project=. -e 'using Eidolon; cfg = load_world("data/seeds/baseline.json"); start_dashboard(cfg)'` opens a blank page at `localhost:8080`.

---

### T3 — Simulation worker
- Implement the `Threads.@spawn` worker loop from ADR-0003 §Worker lifecycle.
- `_pause_flag` / `_reset_flag` atomics; worker polls them at the top of
  each tick and exits cleanly on reset.
- Worker sends `TickSnapshot` to `_snapshots` channel (buffer size: 50).
- No UI wiring yet — test the worker in isolation via unit tests
  (`test/dashboard_tests.jl`).

**Done when:** a test runs the worker for 5 ticks, drains the channel, and
asserts correct `tick` and `mean_opinion` values.

---

### T4 — Start / Pause / Resume / Reset controls
- Wire the `sim_status` reactive field to four Stipple buttons.
- `Start` → spawn worker (guard: only if `sim_status == "idle"`).
- `Pause` → set `_pause_flag = true`.
- `Resume` → set `_pause_flag = false`.
- `Reset` → set `_reset_flag = true`; await worker exit; rebuild ABM; set
  `sim_status = "idle"`.
- Stipple `@periodic 500` timer drains `_snapshots` and updates
  `current_tick`, `mean_opinion`, `opinion_variance` reactive fields.

**Done when:** the browser shows a live tick counter that increments,
pauses when Pause is clicked, and resets to 0 when Reset is clicked.

---

### T5 — Live stats readout
- Display `current_tick / max_ticks`, `mean_opinion` (2 d.p.),
  `opinion_variance` (4 d.p.) on the page.
- Add a simple StipplePlotly line chart: time-series of `mean_opinion`
  updated each timer callback.
- Reuse `opinion_variance_metric` from `src/sensitivity.jl` to keep the
  computation consistent with Phase 4.

**Done when:** the line chart updates in real-time and shows a visible
convergence curve on the baseline seed.

---

### T6 — Network graph visualisation
- Extract the `GraphSpace` graph from the running ABM as `(node_positions,
  edge_list, node_colors)`. Node color maps opinion `[0, 1]` to a
  diverging colorscale (blue–white–red).
- Render with `StipplePlotly` scatter + line traces (nodes + edges).
- Update node colors each timer callback (positions are fixed after init).
- For ≤ 200 agents use a spring layout (Graphs.jl `spring_layout` or a
  precomputed static layout); beyond that down-sample to 200 nodes.

**Done when:** the graph renders the baseline 50-agent network and node
colors shift visibly as opinions converge.

---

### T7 — Broadcast intervention
- Add a text input (`broadcast_text`) and a "Broadcast" button to the UI.
- Handler: append `Intervention(current_tick + 1, "broadcast", Dict("message" => text))`
  to the running ABM's `cfg.interventions`.
- The worker's `_global_event` call already reads interventions every tick,
  so no further wiring is needed.
- Clear `broadcast_text` after submission; show the injected message in a
  small event log on the page.

**Done when:** clicking Broadcast mid-run injects the message and the
event log confirms it, without crashing the simulation.

---

### T8 — Polish and definition of done
- `start_dashboard` prints a `@info` line with the URL.
- Add a minimal `scripts/start_dashboard.jl` entry point (mirrors
  `scripts/run_simulation.jl`).
- Update the roadmap checkboxes in `CLAUDE.md`.
- Smoke test: `test/dashboard_tests.jl` covers worker lifecycle (T3) and
  broadcast injection safety (T7). No browser automation required.

**Done when:** all roadmap Phase 5 bullets are checked, the test suite
passes, and a non-Julia user can follow a three-command README snippet to
open the dashboard.

---

## Definition of done (roadmap)

> A non-Julia user can launch a sim and broadcast an event from the browser.

Specifically:
1. `julia --project=. scripts/start_dashboard.jl` opens the UI.
2. Clicking Start runs the baseline seed and the network graph animates.
3. Typing a message and clicking Broadcast injects it mid-run.
4. All existing tests still pass (`Pkg.test()`).

---

## Open questions

| # | Question | Blocking? | Owner |
|---|----------|-----------|-------|
| Q1 | Does `@periodic` in Stipple work in the same version pinned in Manifest? Verify API before T4. | Yes (T4) | — |
| Q2 | Spring layout for ≤ 200 nodes: use `NetworkLayout.jl` or a manual Fruchterman–Reingold in Graphs.jl? Check what's already transitive. | No (T6) | — |
| Q3 | `cfg.interventions` mutation in T7: add `ReentrantLock` now, or defer until multi-user becomes real? | No (T7) | — |
| Q4 | Stipple pre-1.0 semver: the compat `"0"` bound allows breaking changes. Pin to `"0.X"` once resolved version is known. | No (T1) | — |
