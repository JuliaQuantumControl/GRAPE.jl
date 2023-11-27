using Test
using SafeTestsets
using Plots

unicodeplots()
ENV["GRAPE_LINESEARCH_ANALYSIS_VERBOSE"] = "1"

include("generate_example_tests.jl")

include("download_dumps.jl")

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "GRAPE.jl Package" begin

    println("\n* Pulse Optimization (test_pulse_optimization.jl)")
    @time @safetestset "Pulse Optimization" begin
        include("test_pulse_optimization.jl")
    end

    println("\n* Empty Optimization (test_empty_optimization.jl)")
    @time @safetestset "Empty Optimization" begin
        include("test_empty_optimization.jl")
    end

    println("\n* Taylor Gradient (test_taylor_grad.jl):")
    @time @safetestset "Taylor Gradient" begin
        include("test_taylor_grad.jl")
    end

    println("\n* Example 1 (examples/simple_state_to_state.jl):")
    @time @safetestset "Example 1 (simple_state_to_state)" begin
        include(joinpath("examples", "simple_state_to_state.jl"))
    end

    println("\n* Example 2 (examples/perfect_entanglers.jl):")
    @time @safetestset "Example 2 (perfect_entanglers)" begin
        include(joinpath("examples", "perfect_entanglers.jl"))
    end

    println("")

end
