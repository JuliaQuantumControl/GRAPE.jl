# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using StableRNGs
using LinearAlgebra: norm
using QuantumControl: optimize
using QuantumControl.Functionals: J_T_ss
using QuantumControl.DummyOptimization: dummy_control_problem
using QuantumPropagators: ExpProp
using GRAPE
using GRAPE: step_width, search_direction, pulse_update, gradient
import Optim
import LineSearches


function optim_problem(; kwargs...)
    rng = StableRNG(3143161815)
    return dummy_control_problem(;
        N = 4,
        density = 1.0,
        complex_operators = false,
        rng,
        J_T = J_T_ss,
        prop_method = ExpProp,
        print_iters = false,
        rethrow_exceptions = true,
        kwargs...
    )
end


# A callback that records whether the workspace is consistent with the state of
# the Optim.jl optimizer
function make_check_workspace(; step_is_α_s)
    pulsevals_prev = Float64[]
    gradient_prev = Float64[]
    function check_workspace(wrk, iter)
        state = wrk.optimizer_state
        pulsevals_ok = (wrk.pulsevals == state.x)
        J_ok = (sum(wrk.J_parts) ≈ state.f_x)
        grad_ok = (gradient(wrk; which = :final) ≈ state.g_x)
        if iter == 0
            guess_ok = (wrk.pulsevals_guess == wrk.pulsevals)
            guess_grad_ok = (gradient(wrk) == state.g_x)
            step_ok = true
        else
            guess_ok = (wrk.pulsevals_guess == pulsevals_prev)
            guess_grad_ok = (gradient(wrk) == gradient_prev)
            if step_is_α_s
                Δu = pulse_update(wrk)
                α = step_width(wrk)
                s = search_direction(wrk)
                step_ok = (norm(Δu - α * s) < 1e-10 * norm(Δu))
            else
                step_ok = (search_direction(wrk) == -gradient(wrk))
            end
        end
        pulsevals_prev = copy(wrk.pulsevals)
        gradient_prev = copy(state.g_x)
        checks = (pulsevals_ok, J_ok, grad_ok, guess_ok, guess_grad_ok, step_ok)
        return (iter, wrk.result.J_T, checks)
    end
    return check_workspace
end


@testset "Optim.jl optimizers" begin

    LS = LineSearches
    # name => (optimizer, step_is_α_s, monotonic)
    optimizers = [
        "LBFGS" => (Optim.LBFGS(), true, true),
        "LBFGS (HagerZhang)" => (
            Optim.LBFGS(;
                alphaguess = LS.InitialStatic(alpha = 0.2),
                linesearch = LS.HagerZhang(alphamax = 100.0)
            ),
            true,
            true
        ),
        "LBFGS (BackTracking)" =>
            (Optim.LBFGS(linesearch = LS.BackTracking()), true, true),
        "LBFGS (MoreThuente)" =>
            (Optim.LBFGS(linesearch = LS.MoreThuente()), true, true),
        "BFGS" => (Optim.BFGS(), true, true),
        "GradientDescent" => (Optim.GradientDescent(), true, true),
        "ConjugateGradient" => (Optim.ConjugateGradient(), false, true),
        "MomentumGradientDescent" => (Optim.MomentumGradientDescent(), false, false),
    ]

    for (name, (optimizer, step_is_α_s, monotonic)) in optimizers
        @testset "$name" begin
            problem = optim_problem(;
                optimizer,
                iter_stop = 5,
                allow_f_increases = !monotonic,
                callback = make_check_workspace(; step_is_α_s),
            )
            res = optimize(problem; method = GRAPE)
            @test res.converged
            @test res.message == "Reached maximum number of iterations"
            @test res.iter == 5
            @test [record[1] for record in res.records] == collect(0:5)
            for (iter, _, checks) in res.records
                @test (iter, checks) == (iter, ntuple(_ -> true, length(checks)))
            end
            J_T_vals = [record[2] for record in res.records]
            @test J_T_vals[end] < J_T_vals[1]
            if monotonic
                @test issorted(J_T_vals; rev = true)
            end
            @test res.optimized_controls[1] ≉ res.guess_controls[1]
        end
    end

end


@testset "Optim.jl continue_from" begin
    problem = optim_problem(; optimizer = Optim.LBFGS())
    res1 = optimize(problem; method = GRAPE, iter_stop = 3)
    @test res1.iter == 3
    J_T_1 = res1.J_T
    res2 = optimize(
        problem;
        method = GRAPE,
        iter_stop = 5,
        continue_from = res1,
        callback = (wrk, iter) -> (iter,)
    )
    @test res2.iter == 5
    @test [record[1] for record in res2.records] == [0, 4, 5]
    @test res2.J_T < J_T_1
end


@testset "Optim.jl convergence check" begin
    problem = optim_problem(;
        optimizer = Optim.LBFGS(),
        iter_stop = 100,
        check_convergence = res -> ((res.iter >= 3) && "stop after 3 iterations"),
    )
    res = optimize(problem; method = GRAPE)
    @test res.converged
    @test res.iter == 3
    @test res.message == "stop after 3 iterations"
end


@testset "Optim.jl termination" begin
    # A very large `g_abstol` makes Optim.jl stop before the first iteration
    problem = optim_problem(; optimizer = Optim.LBFGS(), iter_stop = 5, g_abstol = 1e10)
    res = @test_logs (:warn, "Optimization failed to converge") begin
        optimize(problem; method = GRAPE)
    end
    @test !res.converged
    @test res.iter == 0
    @test res.message == "Optim.jl terminated: GradientNorm"
end


@testset "Optim.jl tolerance options" begin
    # Optim.jl deprecates `x_tol`, `f_tol`, and `g_tol`. GRAPE translates them.
    problem = optim_problem(;
        optimizer = Optim.LBFGS(),
        iter_stop = 2,
        x_tol = 0.0,
        f_tol = 0.0,
        g_tol = 1e-8,
    )
    res = @test_logs optimize(problem; method = GRAPE)
    @test res.converged
    @test res.iter == 2
end


@testset "Optim.jl invalid optimizers" begin
    for optimizer in (Optim.NelderMead(), Optim.Newton())
        problem = optim_problem(; optimizer, iter_stop = 2)
        @test_throws ArgumentError optimize(problem; method = GRAPE)
    end
    problem = optim_problem(; optimizer = Optim.LBFGS(), iter_stop = 2, upper_bound = 1.0)
    @test_throws "bounds are not implemented" optimize(problem; method = GRAPE)
end


@testset "Optim.jl callback modifying pulsevals" begin
    problem = optim_problem(;
        optimizer = Optim.LBFGS(),
        iter_stop = 2,
        callback = (wrk, iter) -> (wrk.pulsevals .*= 0.9; nothing),
    )
    @test_throws "must not modify `pulsevals`" optimize(problem; method = GRAPE)
    res = optimize(problem; method = GRAPE, rethrow_exceptions = false)
    @test !res.converged
    @test contains(res.message, "must not modify `pulsevals`")
end
