#!/usr/bin/env julia
# Tiny CLI around `Eidolon.dump_run` / `Eidolon.list_runs`. ADR-0004
# §"Open follow-ups": let a human peek at a stored run without opening a
# SQL prompt.
#
# Usage:
#     julia --project=. scripts/inspect_run.jl <run_id>
#     julia --project=. scripts/inspect_run.jl --list
#     julia --project=. scripts/inspect_run.jl --root /tmp/runs <run_id>
#     julia --project=. scripts/inspect_run.jl --help

using Eidolon

const USAGE = """
Usage:
  julia --project=. scripts/inspect_run.jl <run_id>          dump summary for a run
  julia --project=. scripts/inspect_run.jl --list            list run ids under runs root
  julia --project=. scripts/inspect_run.jl --root <path> ... override EIDOLON_RUNS_ROOT
  julia --project=. scripts/inspect_run.jl --help            show this help
"""

struct CliError <: Exception
    msg::String
end

Base.showerror(io::IO, e::CliError) = print(io, "inspect_run: ", e.msg)

function parse_args(argv::AbstractVector{<:AbstractString})
    help = false
    list = false
    root = nothing
    run_id = nothing
    i = 1
    n = length(argv)
    while i <= n
        a = argv[i]
        if a == "--help" || a == "-h"
            help = true
            i += 1
        elseif a == "--list"
            list = true
            i += 1
        elseif a == "--root"
            i + 1 <= n ||
                throw(CliError("--root requires a path argument"))
            root = argv[i + 1]
            i += 2
        elseif startswith(a, "--")
            throw(CliError("unknown flag: $a"))
        else
            run_id === nothing ||
                throw(CliError("unexpected positional argument: $a"))
            run_id = a
            i += 1
        end
    end
    return (; help, list, root, run_id)
end

function _dispatch(parsed)
    if parsed.list
        for id in Eidolon.list_runs()
            println(id)
        end
        return 0
    end
    if parsed.run_id === nothing
        print(stderr, USAGE)
        return 2
    end
    try
        Eidolon.dump_run(parsed.run_id)
    catch err
        showerror(stderr, err)
        println(stderr)
        return 1
    end
    return 0
end

function main(argv::AbstractVector{<:AbstractString})
    if isempty(argv)
        print(stderr, USAGE)
        return 2
    end
    parsed = try
        parse_args(argv)
    catch err
        if err isa CliError
            println(stderr, "inspect_run: ", err.msg)
            print(stderr, USAGE)
            return 2
        end
        rethrow(err)
    end
    if parsed.help
        print(stderr, USAGE)
        return 2
    end
    if parsed.root === nothing
        return _dispatch(parsed)
    end
    return withenv("EIDOLON_RUNS_ROOT" => parsed.root) do
        _dispatch(parsed)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
