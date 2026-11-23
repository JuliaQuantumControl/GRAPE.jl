# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl:
    hamiltonian, optimize, ControlProblem, Trajectory, propagate_trajectory
using QuantumControl.Functionals: J_T_sm
using QuantumPropagators: ExpProp
using GRAPE
using GRAPE: GrapeWrk, evaluate_gradient!
using LinearAlgebra: norm
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


# The states at final time for the given `control`
function final_states(control)
    H = hamiltonian(σz, (σx, control))
    return [
        propagate_trajectory(Trajectory(traj.initial_state, H), tlist; method = ExpProp) for
        traj in trajectories
    ]
end


@testset "result.states ($gradient_method, $name)" for (gradient_method, name, optimizer) in
                                                       (
    (:gradgen, "L-BFGS-B", nothing),
    (:taylor, "L-BFGS-B", nothing),
    (:gradgen, "Optim.LBFGS", Optim.LBFGS()),
)
    # `result.states` and the state of the forward propagators in a `callback`
    # are the states at final time for the current (optimized) pulses
    guess_states = []
    function callback(wrk, iter)
        if iter == 0
            for propagator in wrk.fw_propagators
                push!(guess_states, copy(propagator.state))
            end
        end
        return nothing
    end
    kw = isnothing(optimizer) ? (;) : (; optimizer)
    problem = ControlProblem(
        trajectories,
        tlist;
        J_T = J_T_sm,
        prop_method = ExpProp,
        iter_stop = 3,
        gradient_method,
        callback,
        kw...
    )
    res = optimize(problem; method = GRAPE, print_iters = false)
    @test res.iter == 3
    @test guess_states ≈ final_states(res.guess_controls[1])
    @test res.states ≈ final_states(res.optimized_controls[1])
    @test !(res.states[1] ≈ trajectories[1].initial_state)
end


# The largest deviation between two vectors of states
max_error(states, expected) = maximum(norm(Ψ - ϕ) for (Ψ, ϕ) in zip(states, expected))


@testset "forward states during backward propagation ($gradient_method)" for (
    gradient_method,
    callback_key,
) in (
    (:gradgen, :grad_prop_callback),
    (:taylor, :bw_prop_callback)
)
    # The backward propagation does not modify the state of the forward
    # propagators, not even temporarily
    expected = final_states(guess)
    wrk_ref = Ref{Any}(nothing)
    seen = []
    function bw_callback(propagator, observables)
        push!(seen, [copy(p.state) for p in wrk_ref[].fw_propagators])
        return nothing
    end
    problem = ControlProblem(
        trajectories,
        tlist;
        J_T = J_T_sm,
        prop_method = ExpProp,
        gradient_method,
        (callback_key => bw_callback,)...
    )
    wrk = GrapeWrk(problem)
    wrk_ref[] = wrk
    G = zeros(length(wrk.pulsevals))
    evaluate_gradient!(G, wrk.pulsevals, wrk)
    @test length(seen) == length(trajectories) * N_T
    @test maximum(max_error(states, expected) for states in seen) < 1e-12
    @test max_error([p.state for p in wrk.fw_propagators], expected) < 1e-12
end
