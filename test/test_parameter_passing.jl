# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl: hamiltonian, optimize, ControlProblem, Trajectory
using QuantumControl.Functionals: J_T_sm
using QuantumPropagators: ExpProp, AbstractPropagator
import QuantumPropagators: init_prop
using GRAPE
using GRAPE: GrapeWrk


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
