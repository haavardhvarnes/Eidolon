module Eidolon

include("io.jl")
include("brain.jl")
include("agents.jl")
include("world.jl")
include("store.jl")

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
       run_simulation,
       open_store,
       record_meta,
       flush_tick!,
       dump_run,
       list_runs

end # module
