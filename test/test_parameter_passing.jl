# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl: hamiltonian, optimize, ControlProblem, Trajectory
using QuantumControl.Controls: discretize_on_midpoints
using QuantumControl.Functionals: J_T_sm
using QuantumPropagators: ExpProp, AbstractPropagator
import QuantumPropagators: init_prop
using GRAPE
using GRAPE: GrapeWrk, step_width, search_direction
import Optim


function tls_problem(; N_T = 20, guess = 0.2, kwargs...)
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    tlist = collect(range(0, 5; length = N_T + 1))
    H = hamiltonian(σz, (σx, fill(guess, N_T)))
    traj = Trajectory(ComplexF64[1, 0], H; target_state = ComplexF64[0, 1])
    return ControlProblem(
        [traj],
        tlist;
        J_T = J_T_sm,
        prop_method = ExpProp,
        print_iters = false,
        kwargs...
    )
end


# Run an optimization and return the result, the list of iteration numbers
# passed to the callback, and the workspace (in its final state)
function optimize_and_record(problem; kwargs...)
    iterations = Int[]
    workspaces = []
    function callback(wrk, iter)
        push!(iterations, iter)
        isempty(workspaces) && push!(workspaces, wrk)
        return nothing
    end
    res = optimize(problem; method = GRAPE, callback, rethrow_exceptions = true, kwargs...)
    return res, iterations, workspaces[1]
end


# A propagation method that does not produce a piecewise propagator
struct NonPiecewiseMethod end

struct NonPiecewisePropagator <: AbstractPropagator end

function init_prop(state, generator, tlist, method::NonPiecewiseMethod; kwargs...)
    return NonPiecewisePropagator()
end


function test_rejects_non_piecewise(problem)
    @test_logs (:error, r"Cannot initialize propagation") match_mode = :any begin
        @test_throws "Cannot initialize propagator as piecewise" GrapeWrk(problem)
    end
end


@testset "piecewise propagators" begin
    # GRAPE requires piecewise propagators for all propagations
    wrk = GrapeWrk(tls_problem())
    @test wrk.fw_prop_kwargs[1][:piecewise] == true
    @test wrk.bw_grad_prop_kwargs[1][:piecewise] == true
    wrk = GrapeWrk(tls_problem(; gradient_method = :taylor))
    @test wrk.fw_prop_kwargs[1][:piecewise] == true
    @test wrk.bw_prop_kwargs[1][:piecewise] == true
    # forward propagation
    test_rejects_non_piecewise(tls_problem(; fw_prop_method = NonPiecewiseMethod()))
    # backward propagation of the gradient generator
    test_rejects_non_piecewise(tls_problem(; grad_prop_method = NonPiecewiseMethod()))
    # backward propagation for `gradient_method = :taylor`
    test_rejects_non_piecewise(
        tls_problem(; gradient_method = :taylor, bw_prop_method = NonPiecewiseMethod())
    )
    # The requirement cannot be disabled
    test_rejects_non_piecewise(
        tls_problem(;
            prop_method = NonPiecewiseMethod(),
            prop_piecewise = false,
            fw_prop_piecewise = false,
            bw_prop_piecewise = false,
            grad_prop_piecewise = false,
        )
    )
end


@testset "enforced propagator arguments" begin
    # The `backward` and `parameters` arguments cannot be overridden
    problem = tls_problem(; grad_prop_backward = false, fw_prop_parameters = IdDict())
    wrk = GrapeWrk(problem)
    @test wrk.bw_grad_prop_kwargs[1][:backward] == true
    ϵ = wrk.controls[1]
    @test parent(wrk.fw_prop_kwargs[1][:parameters][ϵ]) === wrk.pulsevals
    wrk = GrapeWrk(tls_problem(; gradient_method = :taylor, bw_prop_backward = false))
    @test wrk.bw_prop_kwargs[1][:backward] == true
end


@testset "propagation callback observables" begin
    # Each propagation passes its own `observables` to the `prop_callback`
    seen = Dict{Symbol,Vector{Any}}(:fw => [], :bw => [])
    function prop_callback(propagator, observables)
        push!(seen[propagator.backward ? :bw : :fw], observables)
        return nothing
    end
    problem = tls_problem(;
        iter_stop = 1,
        prop_callback,
        fw_prop_observables = :FW,
        grad_prop_observables = :GRAD,
    )
    optimize(problem; method = GRAPE, rethrow_exceptions = true)
    @test unique(seen[:fw]) == [:FW]
    @test unique(seen[:bw]) == [:GRAD]
end


@testset "callback = nothing" begin
    problem = tls_problem(; iter_stop = 2, callback = nothing)
    res = optimize(problem; method = GRAPE)
    @test res.iter == 2
    @test res.message == "Reached maximum number of iterations"
    res = GRAPE.optimize(problem.trajectories, problem.tlist; problem.kwargs...)
    @test res.iter == 2
    @test res.message == "Reached maximum number of iterations"
end


@testset "iter_stop" begin
    for optimizer in (nothing, Optim.LBFGS())
        kw = isnothing(optimizer) ? (;) : (; optimizer)
        # `iter_stop` not larger than the starting iteration: only the guess
        for (iter_start, iter_stop) in ((0, 0), (5, 5), (5, 3))
            problem = tls_problem(; iter_start, iter_stop, kw...)
            res, iterations, _ = optimize_and_record(problem)
            @test iterations == [0]
            @test res.iter == iter_start
            @test res.message == "Reached maximum number of iterations"
        end
        problem = tls_problem(; iter_start = 2, iter_stop = 4, kw...)
        res, iterations, _ = optimize_and_record(problem)
        @test iterations == [0, 3, 4]
        @test res.iter == 4
        # Continuing an optimization (from a copy of `res1`, since
        # `continue_from` updates the result in place)
        res1 = optimize(tls_problem(; iter_stop = 3, kw...); method = GRAPE)
        @test res1.iter == 3
        problem = tls_problem(; iter_stop = 3, continue_from = deepcopy(res1), kw...)
        res, iterations, _ = optimize_and_record(problem)
        @test iterations == [0]
        @test res.iter == 3
        problem = tls_problem(; iter_stop = 5, continue_from = deepcopy(res1), kw...)
        res, iterations, _ = optimize_and_record(problem)
        @test iterations == [0, 4, 5]
        @test res.iter == 5
    end
end


@testset "callback iteration number for the guess" begin
    # The guess is always iteration 0 for the `callback`, and
    # `wrk.result.iter` is the actual starting iteration
    start_iters = Int[]
    function callback(wrk, iter)
        (iter == 0) && push!(start_iters, wrk.result.iter)
        return nothing
    end
    res1 = optimize(tls_problem(; iter_start = 2, iter_stop = 3, callback); method = GRAPE)
    @test start_iters == [2]
    res2 = optimize(
        tls_problem(; iter_stop = 4, continue_from = res1, callback);
        method = GRAPE
    )
    @test start_iters == [2, 3]
    @test res2.iter == 4
end


@testset "guess outside of bounds" begin
    # L-BFGS-B projects the guess into the bounds before the first evaluation
    # of the functional. The workspace must be consistent with that.
    problem = tls_problem(; guess = 0.3, upper_bound = 0.21, iter_stop = 2)
    records = []
    function callback(wrk, iter)
        if iter <= 1
            push!(
                records,
                (
                    pulsevals = copy(wrk.pulsevals),
                    pulsevals_guess = copy(wrk.pulsevals_guess),
                    update = step_width(wrk) * search_direction(wrk),
                )
            )
        end
        return nothing
    end
    msg = "The guess pulses violate the bounds. L-BFGS-B uses the guess projected into the bounds."
    res = @test_logs (:warn, msg) match_mode = :any optimize(
        problem;
        method = GRAPE,
        callback,
        rethrow_exceptions = true
    )
    @test res.iter == 2
    @test all(records[1].pulsevals .== 0.21)
    @test records[1].pulsevals_guess == records[1].pulsevals
    Δϵ = records[2].pulsevals - records[2].pulsevals_guess
    @test Δϵ ≈ records[2].update
end


@testset "continue with active bounds" begin
    # Pulse values that ended on a bound can come back from `continue_from`
    # outside of the bound by round-off. This must not trigger a warning.
    bounds = (; upper_bound = 0.15, lower_bound = -0.05)
    problem = tls_problem(; N_T = 200, guess = 0.1, iter_stop = 10, bounds...)
    res1 = optimize(problem; method = GRAPE)
    pulsevals = discretize_on_midpoints(res1.optimized_controls[1], res1.tlist)
    @test maximum(pulsevals) > bounds.upper_bound  # by round-off
    @test maximum(pulsevals) ≈ bounds.upper_bound
    problem = tls_problem(;
        N_T = 200,
        guess = 0.1,
        iter_stop = 12,
        continue_from = res1,
        bounds...
    )
    res = @test_logs (:info, "Continuing previous optimization") optimize(
        problem;
        method = GRAPE
    )
    @test res.iter == 12
end
