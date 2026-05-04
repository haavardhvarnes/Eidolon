module Eidolon

include("io.jl")
include("agents.jl")
include("world.jl")
include("brain.jl")

export AgentPersona,
    GraphTopology,
    Intervention,
    WorldConfig,
    SchemaError,
    load_world

end # module
