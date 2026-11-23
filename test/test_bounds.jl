# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl: hamiltonian, optimize, ControlProblem, Trajectory
using QuantumControl.Functionals: J_T_sm
using GRAPE
using GRAPE: GrapeWrk


# Two-level system with two controls `ϵ1`, `ϵ2` on `N_T` time intervals
function two_control_problem(; N_T = 4, T = 1.0, kwargs...)
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    tlist = collect(range(0, T; length = N_T + 1))
    ϵ1 = fill(0.1, N_T)
    ϵ2 = fill(0.2, N_T)
    H = hamiltonian(0.5σz, (σx, ϵ1), (σy, ϵ2))
    traj = Trajectory(ComplexF64[1, 0], H; target_state = ComplexF64[0, 1])
    problem = ControlProblem([traj], tlist; J_T = J_T_sm, prop_method = :expprop, kwargs...)
    return problem, ϵ1, ϵ2
end


# Run an optimization and return the result and the workspace (which is in its
# final state after the optimization)
function optimize_with_workspace(problem)
    workspaces = []
    function callback(wrk, iter)
        (iter == 0) && push!(workspaces, wrk)
        return nothing
    end
    res = optimize(
        problem;
        method = GRAPE,
        callback,
        print_iters = false,
        rethrow_exceptions = true
    )
    return res, workspaces[1]
end


@testset "per-control bounds" begin
    # https://github.com/JuliaQuantumControl/GRAPE.jl/issues/121
    problem, ϵ1, ϵ2 = two_control_problem()
    problem.kwargs[:pulse_options] = IdDict(
        ϵ1 =>
            Dict(:lower_bounds => [0.0, 0.1, 0.2, 0.3], :upper_bounds => fill(1.0, 4)),
        ϵ2 =>
            Dict(:lower_bounds => fill(-5.0, 4), :upper_bounds => [5.0, 6.0, 7.0, 8.0]),
    )
    wrk = GrapeWrk(problem)
    @test wrk.lower_bounds == [0.0, 0.1, 0.2, 0.3, -5.0, -5.0, -5.0, -5.0]
    @test wrk.upper_bounds == [1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]
end


@testset "per-control bounds fallback" begin
    # Infinite values in `pulse_options` fall back to the global bounds
    problem, ϵ1, ϵ2 = two_control_problem(; upper_bound = 2.0, lower_bound = -2.0)
    problem.kwargs[:pulse_options] = IdDict(
        ϵ1 => Dict(:upper_bounds => [1.0, Inf, 1.0, Inf]),
        ϵ2 => Dict(:lower_bounds => [-Inf, -1.0, -Inf, -1.0]),
    )
    wrk = GrapeWrk(problem)
    @test wrk.upper_bounds == [1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    @test wrk.lower_bounds == [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -1.0]
    # Without global bounds, infinite values mean "unbounded"
    problem, ϵ1, ϵ2 = two_control_problem()
    problem.kwargs[:pulse_options] =
        IdDict(ϵ1 => Dict(:upper_bounds => [1.0, Inf, 1.0, Inf]))
    wrk = GrapeWrk(problem)
    @test wrk.upper_bounds == [1.0, Inf, 1.0, Inf, Inf, Inf, Inf, Inf]
    @test all(wrk.lower_bounds .== -Inf)
end


@testset "per-control bounds scalar" begin
    # A single number applies to all time intervals of that control. Integer
    # values are converted to Float64, also for the global bounds. Finite
    # per-control bounds take precedence over the global bounds, even if they
    # are less restrictive.
    problem, ϵ1, ϵ2 = two_control_problem(; upper_bound = 2)
    problem.kwargs[:pulse_options] = IdDict(
        ϵ1 => Dict(:upper_bounds => 3, :lower_bounds => -0.5),
        ϵ2 => Dict(:upper_bounds => [1.5, Inf, 1.5, Inf]),
    )
    wrk = GrapeWrk(problem)
    @test wrk.upper_bounds == [3.0, 3.0, 3.0, 3.0, 1.5, 2.0, 1.5, 2.0]
    @test wrk.lower_bounds == [-0.5, -0.5, -0.5, -0.5, -Inf, -Inf, -Inf, -Inf]
end


@testset "per-control bounds invalid" begin
    problem, ϵ1, ϵ2 = two_control_problem()
    problem.kwargs[:pulse_options] = IdDict(ϵ1 => Dict(:upper_bounds => fill(1.0, 3)))
    msg = "`pulse_options[control][:upper_bounds]` for control 1 must be a number or have one value for each of the 4 time intervals, not 3"
    @test_throws ArgumentError(msg) GrapeWrk(problem)
    problem.kwargs[:pulse_options] = IdDict(ϵ2 => Dict(:lower_bounds => [0.0]))
    msg = "`pulse_options[control][:lower_bounds]` for control 2 must be a number or have one value for each of the 4 time intervals, not 1"
    @test_throws ArgumentError(msg) GrapeWrk(problem)
end


@testset "per-control bounds optimization" begin
    # The bounds of each control are enforced on that control only. Without
    # bounds, both controls reach values of about ±0.9.
    N_T = 20
    problem, ϵ1, ϵ2 = two_control_problem(; N_T, T = 5.0, iter_stop = 20)
    problem.kwargs[:pulse_options] = IdDict(
        ϵ1 => Dict(:lower_bounds => fill(0.0, N_T), :upper_bounds => fill(0.15, N_T)),
        ϵ2 => Dict(:lower_bounds => fill(-0.5, N_T), :upper_bounds => fill(0.5, N_T)),
    )
    res, wrk = optimize_with_workspace(problem)
    @test res.J_T < 1e-10
    ϵ1_vals = wrk.pulsevals[1:N_T]
    ϵ2_vals = wrk.pulsevals[(N_T+1):(2N_T)]
    @test all(0.0 .<= ϵ1_vals .<= 0.15)
    @test all(-0.5 .<= ϵ2_vals .<= 0.5)
    # The bounds are binding: L-BFGS-B projects values onto the bounds
    is_on_bound(vals, lb, ub) = any((vals .== lb) .| (vals .== ub))
    @test is_on_bound(ϵ1_vals, 0.0, 0.15)
    @test is_on_bound(ϵ2_vals, -0.5, 0.5)
    ϵ1_opt, ϵ2_opt = res.optimized_controls
    @test all(0.0 .<= ϵ1_opt .<= 0.15)
    @test all(-0.5 .<= ϵ2_opt .<= 0.5)
end
