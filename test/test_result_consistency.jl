# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl:
    hamiltonian, optimize, ControlProblem, Trajectory, propagate_trajectory
using QuantumControl.Controls: discretize
using QuantumControl.Functionals: J_T_sm
using QuantumPropagators: ExpProp
using GRAPE
using GRAPE: evaluate_functional
import Krotov
using LinearAlgebra: dot
import Optim


σx = ComplexF64[0 1; 1 0]
σz = ComplexF64[1 0; 0 -1]
N_T = 20
tlist = collect(range(0, 5; length = N_T + 1))
guess = fill(0.2, N_T)
H = hamiltonian(σz, (σx, guess))
trajectories = [
    Trajectory(ComplexF64[1, 0], H; target_state = ComplexF64[0, 1]),
    Trajectory(ComplexF64[0, 1], H; target_state = ComplexF64[1, 0]),
]

optimizers = (("L-BFGS-B", nothing), ("Optim.LBFGS", Optim.LBFGS()))


# The states at final time for the given `control`
function final_states(control)
    H = hamiltonian(σz, (σx, control))
    return [
        propagate_trajectory(Trajectory(traj.initial_state, H), tlist; method = ExpProp) for
        traj in trajectories
    ]
end


# Whether all attributes of `res` describe the same pulses, namely
# `res.optimized_controls` (with an absolute tolerance for values close to zero,
# e.g., `J_T` near convergence)
function is_consistent(res; atol = 1e-12)
    states = final_states(res.optimized_controls[1])
    tau_vals = [dot(traj.target_state, Ψ) for (traj, Ψ) in zip(trajectories, states)]
    return (
        isapprox(res.states, states; atol) &&
        isapprox(res.tau_vals, tau_vals; atol) &&
        isapprox(res.J_T, J_T_sm(states, trajectories); atol)
    )
end


function make_problem(; optimizer = nothing, kwargs...)
    kw = isnothing(optimizer) ? (;) : (; optimizer)
    return ControlProblem(
        trajectories,
        tlist;
        J_T = J_T_sm,
        prop_method = ExpProp,
        kw...,
        kwargs...
    )
end


@testset "result in callback ($name)" for (name, optimizer) in optimizers
    # In a `callback`, the result describes the current iteration
    consistent = Bool[]
    current_pulses = Bool[]
    function callback(wrk, iter)
        push!(consistent, is_consistent(wrk.result))
        ϵ = discretize(wrk.pulsevals, tlist)
        push!(current_pulses, wrk.result.optimized_controls[1] ≈ ϵ)
        return nothing
    end
    problem = make_problem(; optimizer, iter_stop = 3, callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test res.iter == 3
    @test length(consistent) == 4  # guess and 3 iterations
    @test all(consistent)
    @test all(current_pulses)
    @test is_consistent(res)
end


# A `fw_prop_callback` that throws an exception in the middle of the forward
# propagation of trajectory `k` in evaluation number `n_eval` of the functional
function make_interrupting_callback(n_eval, k)
    n_steps = 0
    function fw_prop_callback(propagator, observables)
        n_steps += 1
        if n_steps == ((n_eval - 1) * length(trajectories) + (k - 1)) * N_T + N_T ÷ 2
            error("Simulated interruption")
        end
        return nothing
    end
    return fw_prop_callback
end


@testset "result after exception ($name, evaluation $n_eval, trajectory $k)" for (
        name,
        optimizer,
    ) in optimizers,
    n_eval in (3, 4, 5, 6),
    k in (1, 2)
    # An exception in the middle of an evaluation of the functional (e.g., from
    # Ctrl-C) leaves the result for the last completed iteration. For `k = 2`,
    # the evaluation has already calculated `tau_vals` for the first
    # trajectory.
    fw_prop_callback = make_interrupting_callback(n_eval, k)
    problem = make_problem(; optimizer, iter_stop = 10, fw_prop_callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test startswith(res.message, "Exception: Simulated interruption")
    @test is_consistent(res)
end


@testset "result after exception in guess ($name)" for (name, optimizer) in optimizers
    # If the optimization ends before the guess has been evaluated, `J_T` and
    # `tau_vals` are `NaN`
    fw_prop_callback = make_interrupting_callback(1, 2)
    problem = make_problem(; optimizer, iter_stop = 10, fw_prop_callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test startswith(res.message, "Exception: Simulated interruption")
    @test res.iter == 0
    @test isnan(res.J_T)
    @test all(isnan, res.tau_vals)
    @test res.optimized_controls == res.guess_controls
end


@testset "result after exception in callback ($name)" for (name, optimizer) in optimizers
    function callback(wrk, iter)
        (iter == 2) && error("Simulated interruption")
        return nothing
    end
    problem = make_problem(; optimizer, iter_stop = 10, callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test startswith(res.message, "Exception: Simulated interruption")
    @test res.iter == 2
    @test is_consistent(res)
end


@testset "callback modifying pulsevals" begin
    # If a `callback` modifies the pulse values, the `optimized_controls`
    # include the modification, while the other attributes are for the pulse
    # values before the modification
    pulsevals_before = Float64[]
    function callback(wrk, iter)
        if iter == 2
            append!(pulsevals_before, wrk.pulsevals)
            wrk.pulsevals .*= 0.9
        end
        return nothing
    end
    problem = make_problem(; iter_stop = 2, callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test res.iter == 2
    @test res.optimized_controls[1] ≈ discretize(0.9 .* pulsevals_before, tlist)
    @test res.states ≈ final_states(discretize(pulsevals_before, tlist))
end


@testset "callback modifying pulsevals (Optim.LBFGS)" begin
    # With an Optim.jl optimizer, modifying the pulse values in a `callback` is
    # an error, and the result is for the pulse values before the modification
    function callback(wrk, iter)
        (iter == 2) && (wrk.pulsevals .*= 0.9)
        return nothing
    end
    problem = make_problem(; optimizer = Optim.LBFGS(), iter_stop = 10, callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test startswith(res.message, "Exception: A `callback` must not modify `pulsevals`")
    @test res.iter == 2
    @test is_consistent(res)
end


@testset "continue_from a GrapeResult ($name)" for (name, optimizer) in optimizers
    # A `GrapeResult` given as `continue_from` is updated in place and remains
    # consistent
    res1 = optimize(
        make_problem(; optimizer, iter_stop = 2);
        method = GRAPE,
        print_iters = false
    )
    problem = make_problem(; optimizer, iter_stop = 4, continue_from = res1)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test res === res1
    @test res.iter == 4
    @test is_consistent(res)
end


@testset "evaluating the functional in a callback ($name)" for (name, optimizer) in
                                                               optimizers
    # Evaluating the functional for other pulse values (e.g., to analyze the
    # line search) does not change the result
    unchanged = Bool[]
    function callback(wrk, iter)
        if iter > 0
            res = wrk.result
            before = deepcopy((res.states, res.tau_vals, res.J_T, res.optimized_controls))
            pulsevals = copy(wrk.pulsevals)
            evaluate_functional(1.5 .* pulsevals, wrk; count_call = false)
            after = (res.states, res.tau_vals, res.J_T, res.optimized_controls)
            push!(unchanged, after == before)
            # Restore the workspace, which the optimizer requires
            evaluate_functional(pulsevals, wrk; count_call = false)
        end
        return nothing
    end
    problem = make_problem(; optimizer, iter_stop = 2, callback)
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test res.iter == 2
    @test unchanged == [true, true]
    @test is_consistent(res)
end


@testset "continue_from a Krotov result" begin
    # Continuing from a result of a different method does not modify that
    # result
    problem = make_problem(; lambda_a = 5, update_shape = (t -> 1.0))
    res_krotov = optimize(problem; method = Krotov, iter_stop = 2, print_iters = false)
    res_krotov_before = deepcopy(res_krotov)
    res = optimize(
        problem;
        method = GRAPE,
        iter_stop = 4,
        continue_from = res_krotov,
        print_iters = false
    )
    @test res.iter == 4
    @test is_consistent(res)
    for field in (:optimized_controls, :states, :tau_vals, :J_T, :iter)
        @test getfield(res_krotov, field) == getfield(res_krotov_before, field)
    end
end
