module Eidolon

include("io.jl")
include("brain.jl")
include("agents.jl")
include("world.jl")
include("store.jl")
include("sweep.jl")
include("sensitivity.jl")
include("dashboard.jl")

export AgentPersona,
       GraphTopology,
       Intervention,
       WorldConfig,
       SchemaError,
       load_world,
       AbstractBrain,
       NullBrain,
       RandomBrain,
       LiveBrain,
       ReplayBrain,
       BrainOutput,
       LLMBudgetExceeded,
       SchemaDriftError,
       ReplayMissingError,
       live_brain,
       replay_brain,
       batch_reflect!,
       EidolonAgent,
       agent_step!,
       initialize_world,
       run_simulation,
       open_store,
       record_meta,
       flush_tick!,
       dump_run,
       list_runs,
       auto_run_id,
       load_trajectories,
       load_transcripts,
       expand_grid,
       grid_sweep,
       SensitivityDimension,
       SensitivityResult,
       opinion_variance_metric,
       persona_sensitivity,
       top_personas,
       TickSnapshot,
       DashboardModel,
       start_dashboard,
       stop_dashboard

end # module
