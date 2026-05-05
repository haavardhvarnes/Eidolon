using Eidolon
using Test

@testset "Eidolon" begin
    @testset "smoke" begin
        @test isdefined(Main, :Eidolon)
    end

    include("io_tests.jl")
    include("world_tests.jl")
    include("store_tests.jl")
    include("dump_run_tests.jl")
    include("brain_tests.jl")
    include("replay_tests.jl")
end
