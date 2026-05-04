module Eidolon

include("io.jl")
include("brain.jl")
include("agents.jl")
include("world.jl")

export AgentPersona,
       GraphTopology,
       Intervention,
       WorldConfig,
       SchemaError,
       load_world,
       AbstractBrain,
       NullBrain,
       EidolonAgent,
       agent_step!,
       initialize_world,
       run_simulation

end # module
