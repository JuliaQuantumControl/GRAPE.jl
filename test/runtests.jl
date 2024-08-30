using Test
using SafeTestsets
using Plots

unicodeplots()

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "GRAPE.jl Package" begin

    println("\n* TLS Optimization (test_tls_optimization.jl)")
    @time @safetestset "TLS Optimization" begin
        include("test_tls_optimization.jl")
    end

    println("\n* Pulse Optimization (test_pulse_optimization.jl)")
    @time @safetestset "Pulse Optimization" begin
        include("test_pulse_optimization.jl")
    end

    println("\n* Empty Optimization (test_empty_optimization.jl)")
    @time @safetestset "Empty Optimization" begin
        include("test_empty_optimization.jl")
    end

    println("\n* Pulse Running Cost (test_pulse_running_cost.jl)")
    @time @safetestset "Pulse Running Cost" begin
        include("test_pulse_running_cost.jl")
    end

    println("\n* Taylor Gradient (test_taylor_grad.jl):")
    @time @safetestset "Taylor Gradient" begin
        include("test_taylor_grad.jl")
    end

    println("\n* Iterations (test_iterations.jl)")
    @time @safetestset "Iterations" begin
        include("test_iterations.jl")
    end

end
nothing
