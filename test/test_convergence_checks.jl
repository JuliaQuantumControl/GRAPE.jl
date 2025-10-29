# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using StableRNGs
using GRAPE
using QuantumControl.Functionals: J_T_ss
using QuantumPropagators: ExpProp
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem
using IOCapture

PASSTHROUGH = false

@testset "convergence check" begin
    rng = StableRNG(1244538994)
    problem = dummy_control_problem(;
        N = 2,
        density = 1.0,
        complex_operators = false,
        rng,
        J_T = J_T_ss,
        prop_method = ExpProp,
        check_convergence = (res -> ((res.J_T < 1e-5) && "J_T < 10⁻⁵")),
        callback = GRAPE.make_grape_print_iters(store_iter_info = ["iter.", "J_T"]),
    )
    captured = IOCapture.capture(passthrough = PASSTHROUGH) do
        GRAPE.optimize(problem.trajectories, problem.tlist; problem.kwargs..., iter_stop = 100)
    end
    res = captured.value
    @test res.converged
    @test res.iter_start == 0
    @test res.iter_stop == 100
    @test res.iter == 17
    @test res.message == "J_T < 10⁻⁵"
end


@testset "convergence check with iter_stop" begin
    rng = StableRNG(1244538994)
    problem = dummy_control_problem(;
        N = 2,
        density = 1.0,
        complex_operators = false,
        rng,
        J_T = J_T_ss,
        prop_method = ExpProp,
        check_convergence = (res -> ((res.J_T < 1e-5) && "J_T < 10⁻⁵")),
        callback = GRAPE.make_grape_print_iters(store_iter_info = ["iter.", "J_T"]),
    )
    captured = IOCapture.capture(passthrough = PASSTHROUGH) do
        GRAPE.optimize(problem.trajectories, problem.tlist; problem.kwargs..., iter_stop = 2)
    end
    res = captured.value
    @test res.converged
    @test res.iter == 2
    @test res.message == "Reached maximum number of iterations"
end
