using Test
using QuantumControl
using LinearAlgebra
using StableRNGs
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem
using QuantumControl.Controls: get_controls, discretize_on_midpoints
using QuantumControl.Functionals: J_T_re

@testset "pulse optimization" begin

    # Test the resolution of
    # https://github.com/JuliaQuantumControl/Krotov.jl/issues/28
    # 
    # While this hasn't been a problem for GRAPE, we'd want to make sure that
    # any future changes won't result in the optimization mutating the guess
    # controls

    rng = StableRNG(1244561944)

    problem = dummy_control_problem(; pulses_as_controls=true)
    nt = length(problem.tlist)
    guess_pulse = QuantumControl.Controls.get_controls(problem.objectives)[1]
    @test length(guess_pulse) == nt - 1
    guess_pulse_copy = copy(QuantumControl.Controls.get_controls(problem.objectives)[1])

    # Optimizing this should not modify the original generator in any way
    res = optimize(problem; method=:GRAPE, J_T=J_T_re, iter_stop=2)
    opt_control = res.optimized_controls[1]
    @test length(opt_control) == nt  # optimized_controls are always *on* tlist
    opt_pulse = discretize_on_midpoints(opt_control, problem.tlist)
    post_pulse = QuantumControl.Controls.get_controls(problem.objectives)[1]

    # * The generator should still have the exact same objects as controls
    @test guess_pulse ≡ post_pulse
    # * These objects should not have been modified
    @test norm(guess_pulse_copy - guess_pulse) ≈ 0.0
    # * But the values of the optimized pulse should differ from the pulse in
    #   the generator
    @test norm(post_pulse - opt_pulse) > 0.1

end
