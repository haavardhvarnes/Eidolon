# Eidolon.jl

A predictive swarm-intelligence simulator written in Julia. Each agent's
reasoning is driven by an LLM "brain" coupled to a small-world social
network; the simulator studies macro-level outcomes (opinion convergence,
polarisation, cascades) and identifies which agent personas drive them.

Eidolon.jl is a Julia port of the [MiroFish](https://github.com/666ghj/MiroFish)
project.

## Status

Pre-alpha — scaffold only. See the development roadmap and architecture
notes in [`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

A reproducible dev environment is available via the
[`.devcontainer/`](.devcontainer/) (VS Code → *Dev Containers: Reopen in
Container*). It pre-installs Claude Code, pins Julia, and applies an egress
firewall so `claude --dangerously-skip-permissions` can be used safely.

## Layout

| Path                  | Purpose                                          |
|-----------------------|--------------------------------------------------|
| `src/`                | Module source (agents, world, brain, io)         |
| `test/`               | Unit tests, run with `Pkg.test()`                |
| `data/seeds/`         | JSON seed files (initial world & personas)       |
| `scripts/`            | Entry points, e.g. `run_simulation.jl`           |
| `runs/`               | Per-run artefacts (DuckDB, transcripts) — ignored|
| `docs/adr/`           | Architecture Decision Records                    |
| `.devcontainer/`      | Dev container with firewalled YOLO mode          |
| [`CLAUDE.md`](CLAUDE.md) | Roadmap, conventions, decisions               |

## License

[MIT](LICENSE).
