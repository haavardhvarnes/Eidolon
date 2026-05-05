#!/usr/bin/env julia
# Launch the Eidolon web dashboard. Loads a seed file, starts the
# Stipple server on the requested port, then blocks until Ctrl-C.
#
# Usage:
#     julia --project=. scripts/start_dashboard.jl
#     julia --project=. scripts/start_dashboard.jl --seed data/seeds/baseline.json
#     julia --project=. scripts/start_dashboard.jl --seed <path> --port 8080
#     julia --project=. scripts/start_dashboard.jl --help

using Eidolon

const USAGE = """
Usage:
  julia --project=. scripts/start_dashboard.jl
  julia --project=. scripts/start_dashboard.jl --seed <path>   load specific seed file
  julia --project=. scripts/start_dashboard.jl --port <n>      listen on port n (default 8080)
  julia --project=. scripts/start_dashboard.jl --help          show this help
"""

const DEFAULT_SEED = joinpath(@__DIR__, "..", "data", "seeds", "baseline.json")
const DEFAULT_PORT = 8080

function parse_args(argv::AbstractVector{<:AbstractString})
    seed = nothing
    port = nothing
    help = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a in ("--help", "-h")
            help = true; i += 1
        elseif a == "--seed"
            i + 1 <= length(argv) || error("--seed requires a path argument")
            seed = argv[i + 1]; i += 2
        elseif a == "--port"
            i + 1 <= length(argv) || error("--port requires a number argument")
            port = parse(Int, argv[i + 1]); i += 2
        else
            error("unknown argument: $a")
        end
    end
    return (; seed = something(seed, DEFAULT_SEED),
              port = something(port, DEFAULT_PORT),
              help)
end

function main(argv)
    parsed = try
        parse_args(argv)
    catch err
        println(stderr, "start_dashboard: ", err.msg)
        print(stderr, USAGE)
        return 2
    end

    if parsed.help
        print(stdout, USAGE)
        return 0
    end

    isfile(parsed.seed) ||
        error("seed file not found: $(parsed.seed)")

    cfg = load_world(parsed.seed)
    start_dashboard(cfg; port = parsed.port)

    # Block until Ctrl-C so the server stays up.
    @info "Press Ctrl-C to stop."
    try
        while true
            sleep(3600)
        end
    catch e
        e isa InterruptException || rethrow(e)
        @info "Shutting down."
        Stipple.down()
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
