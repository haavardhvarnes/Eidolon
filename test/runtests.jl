using Eidolon
using Test

@testset "Eidolon" begin
    @testset "smoke" begin
        @test isdefined(Main, :Eidolon)
    end

    include("io_tests.jl")
end
