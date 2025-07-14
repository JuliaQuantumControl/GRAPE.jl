# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using QuantumControl
using QuantumControl.Functionals: J_a_fluence
using Test
using StableRNGs
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem
using QuantumControl.Functionals: J_T_re
using LinearAlgebra: norm
using GRAPE

@testset "running cost with manual gradient" begin

    function _TEST_J_a_smoothness(pulsevals, tlist)
        N = length(tlist) - 1  # number of intervals
        L = length(pulsevals) ÷ N
        @assert length(pulsevals) == N * L
        J_a = 0.0
        for l = 1:L
            for n = 2:N
                J_a += (pulsevals[(l-1)*N+n] - pulsevals[(l-1)*N+n-1])^2
            end
        end
        return 0.5 * J_a
    end

    function _TEST_grad_J_a_smoothness(pulsevals, tlist)
        N = length(tlist) - 1  # number of intervals
        L = length(pulsevals) ÷ N
        ∇J_a = zeros(length(pulsevals))
        for l = 1:L
            for n = 1:N
                ∇J_a[(l-1)*N+n] = 0.0
                uₙ = pulsevals[(l-1)*N+n]
                if n > 1
                    uₙ₋₁ = pulsevals[(l-1)*N+n-1]
                    ∇J_a[(l-1)*N+n] += (uₙ - uₙ₋₁)
                end
                if n < N
                    uₙ₊₁ = pulsevals[(l-1)*N+n+1]
                    ∇J_a[(l-1)*N+n] += (uₙ - uₙ₊₁)
                end
            end
        end
        return ∇J_a
    end

    rng = StableRNG(1244561944)
    problem = dummy_control_problem(; n_controls=2, rng)
    res = optimize(
        problem;
        method=GRAPE,
        J_a=_TEST_J_a_smoothness,
        grad_J_a=_TEST_grad_J_a_smoothness,
        lambda_a=0.1,
        J_T=J_T_re,
        iter_stop=2
    )
    @test res.converged
    @test res.J_T < res.J_T_prev

end


@testset "J_a_fluence running cost" begin

    rng = StableRNG(1244561944)
    problem = dummy_control_problem(; n_controls=2, rng)
    res0 = optimize(problem; method=GRAPE, J_T=J_T_re, iter_stop=2)
    res = optimize(problem; method=GRAPE, J_a=J_a_fluence, J_T=J_T_re, iter_stop=2)
    @test res0.converged
    @test res.converged
    @test sum(norm.(res.optimized_controls)) < sum(norm.(res0.optimized_controls))

end
