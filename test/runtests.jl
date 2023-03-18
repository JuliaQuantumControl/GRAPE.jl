using Test
using SafeTestsets
using Plots

unicodeplots()
ENV["GRAPE_LINESEARCH_ANALYSIS_VERBOSE"] = "1"

include(joinpath(@__DIR__, "generate_example_tests.jl"))

include(joinpath(@__DIR__, "download_dumps.jl"))

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "GRAPE.jl Package" begin

    print("\n* Pulse Optimization (test_pulse_optimization.jl)")
    @time @safetestset "Pulse Optimization" begin
        include("test_pulse_optimization.jl")
    end

    print("\n* Taylor Gradient (test_taylor_grad.jl):")
    @time @safetestset "Taylor Gradient" begin
        include("test_taylor_grad.jl")
    end

    print("\n* Example 1 (examples/simple_state_to_state.jl):")
    @time @safetestset "Example 1" begin
        include(joinpath("examples", "simple_state_to_state.jl"))
    end

    print("\n* Example 2 (examples/perfect_entanglers.jl):")
    @time @safetestset "Example 2" begin
        include(joinpath("examples", "perfect_entanglers.jl"))
    end

    println("")

end
