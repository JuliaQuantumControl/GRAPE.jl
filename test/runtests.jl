using Test
using SafeTestsets

include("generate_example_tests.jl")

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose=true "Krotov.jl Package" begin

    print("\n* Example 1 (examples/simple_state_to_state.jl):")
    @time @safetestset "Example 1" begin include(joinpath("examples", "simple_state_to_state.jl")) end

    println("")

end
