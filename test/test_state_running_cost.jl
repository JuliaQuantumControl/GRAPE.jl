# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using QuantumControl
using QuantumControl.Functionals: J_T_re, make_xi, J_b
using Test
using StableRNGs
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem
using QuantumPropagators.Interfaces: check_state
using LinearAlgebra: dot, norm
using GRAPE
using Zygote
using IOCapture


@testset "state running cost with manual xi" begin

    rng = StableRNG(1244561944)
    N = 10
    problem = dummy_control_problem(; N, n_controls = 2, rng)
    tlist = problem.tlist
    trajectories = problem.trajectories

    # Build a random Hermitian "penalty" operator D using the initial state
    # as a template
    Ψ₀ = trajectories[1].initial_state
    A = randn(rng, ComplexF64, N, N)
    D = A * A' / N  # positive semidefinite, so g_b >= 0

    # g_b penalizes the expectation value of D
    function g_b(Ψ, traj, tlist, n)
        return real(dot(Ψ, D * Ψ))
    end

    # Analytic xi: -D * Ψ (Wirtinger derivative: -∂g_b/∂⟨Ψ|)
    function xi_manual(Ψ, traj, tlist, n)
        return -D * Ψ
    end

    function check_J_b(wrk, iter)
        g_b_func = wrk.kwargs[:g_b]
        lambda_b = get(wrk.kwargs, :lambda_b, 1.0)
        J_b_val = J_b(wrk.fw_storage, wrk.trajectories, wrk.tlist; g_b = g_b_func)
        @test J_b_val ≈ sum(wrk.J_b_trajectory)
        @test J_b_val * lambda_b ≈ wrk.J_parts[3]
        return nothing
    end

    res = optimize(
        problem;
        method = GRAPE,
        J_T = J_T_re,
        g_b = g_b,
        xi = xi_manual,
        lambda_b = 0.5,
        iter_stop = 5,
        callback = check_J_b,
        rethrow_exceptions = true,
    )
    @test res.converged
    @test res.J_T < 1.0  # some optimization happened
    @test res.J_b >= 0.0  # g_b >= 0 for PSD D, so J_b >= 0

end


@testset "state running cost with auto xi (Zygote)" begin

    import Zygote

    N = 10
    rng = StableRNG(1244561944)
    problem = dummy_control_problem(; N, n_trajectories = 2, n_controls = 2, rng)

    function positive_semidefinite_matrix(N; rng = rng)
        A = randn(rng, ComplexF64, N, N)
        A * A' / N
    end

    trajectories = [
        Trajectory(
            traj.initial_state,
            traj.generator;
            target_state = traj.target_state,
            D = positive_semidefinite_matrix(N; rng)
        ) for traj in problem.trajectories
    ]
    @test norm(trajectories[1].D - trajectories[2].D) > 1e-2
    for (i, traj) in enumerate(trajectories)
        problem.trajectories[i] = traj
    end

    function g_b(Ψ, traj, tlist, n)
        return real(dot(Ψ, traj.D * Ψ))
    end

    Ψ = [copy(traj.initial_state) for traj in trajectories]
    @test J_T_re(Ψ, trajectories) > 0.0

    # Auto-generate xi via Zygote; should match -D * Psi analytically
    xi_auto = make_xi(g_b; automatic = :Zygote)

    tlist = problem.tlist
    ξ1 = xi_auto(Ψ[1], trajectories[1], tlist, 1)
    ξ2 = xi_auto(Ψ[2], trajectories[2], tlist, 1)
    @test check_state(ξ1)
    @test check_state(ξ2)
    @test norm(ξ1 - ξ2) > 1e-2

    λ_b = 1e-3

    captured = IOCapture.capture(passthrough = true) do
        optimize(
            problem;
            method = GRAPE,
            J_T = J_T_re,
            g_b = g_b,
            xi = xi_auto,
            lambda_b = λ_b,
            iter_stop = 5,
            print_iter_info = [
                "iter.",
                "J",
                "J_T",
                "J_b",
                "λ_b⋅J_b",
                "ǁ∇Jǁ",
                "ǁ∇(J_T+λ_b·J_b)ǁ",
                "ǁΔϵǁ",
                "ΔJ"
            ],
            store_iter_info = [
                "J",
                "J_T",
                "J_b",
                "λ_b⋅J_b",
                "ǁ∇(J_T+λ_b·J_b)ǁ",
                "ǁ∇J_Tǁ",
                "ΔJ"
            ],
        )
    end
    res = captured.value
    @test contains(
        captured.output,
        "Warning: The label \"ǁ∇J_Tǁ\" was requested, but the optimization includes a state-dependent running cost `g_b`."
    )
    for (i, (J, J_T, J_b, λ_b_J_b, nrm_grad_Tb, nrm_grad_T, ΔJ)) in enumerate(res.records)
        (i > 1) && @test ΔJ < 0.0
        @test J ≈ J_T + λ_b_J_b
        @test λ_b_J_b ≈ λ_b * J_b
        # Both labels point to the same value (norm of wrk.grad_J_Tb).
        @test nrm_grad_Tb == nrm_grad_T
    end
    @test res.converged
    @test res.J_b >= 0.0

    # Using the "ǁ∇(J_T+λ_b·J_b)ǁ" label without g_b should also emit a
    # warning at iteration 0.
    captured = IOCapture.capture(passthrough = false) do
        optimize(
            problem;
            method = GRAPE,
            J_T = J_T_re,
            iter_stop = 1,
            print_iter_info = ["iter.", "J_T", "ǁ∇(J_T+λ_b·J_b)ǁ"],
        )
    end
    @test contains(captured.output, "Warning: The label \"ǁ∇(J_T+λ_b·J_b)ǁ\" was requested")

    # Verify xi_auto matches the analytic gradient -D * Psi
    D = trajectories[1].D
    xi_analytic = -D * Ψ[1]
    xi_zygote = xi_auto(Ψ[1], trajectories[1], tlist, 1)
    @test norm(xi_zygote - xi_analytic) < 1e-14

end

@testset "STIRAP running-cost optimization" begin

    𝕚 = 1im
    ω₁ = 0.0;
    ω₂ = 10.0;
    ω₃ = 5.0;
    ω_P = 9.5;
    ω_S = 4.5;

    Δ_P = (ω₂ - ω₁) - ω_P;
    Δ_S = (ω₂ - ω₃) - ω_S;

    using LinearAlgebra: Diagonal
    H0 = Array(Diagonal(ComplexF64[0, Δ_P, Δ_P-Δ_S]))

    H1P_re = 0.5 * ComplexF64[
        0  1  0
        1  0  0
        0  0  0
    ]

    H1P_im = 0.5 * ComplexF64[
         0  𝕚  0
        -𝕚  0  0
         0  0  0
    ]

    H1S_re = 0.5 * ComplexF64[
        0  0  0
        0  0  1
        0  1  0
    ]

    H1S_im = 0.5 * ComplexF64[
        0  0  0
        0  0  𝕚
        0 -𝕚  0
    ]

    using QuantumControl.Shapes: blackman

    H = hamiltonian(
        H0,
        (H1P_re, t -> blackman(t, 1.0, 5.0)),
        (H1P_im, t -> 0.0),
        (H1S_re, t -> blackman(t, 0.0, 4.0)),
        (H1S_im, t -> 0.0)
    );

    tlist = collect(range(0, 5; length = 501));

    ket1 = ComplexF64[1, 0, 0]
    ket2 = ComplexF64[0, 1, 0]
    ket3 = ComplexF64[0, 0, 1]
    trajectory = Trajectory(ket1, H; target_state = ket3)

    using QuantumControl.Functionals: J_T_ss
    using QuantumPropagators: Cheby

    function g_b(Ψ, _, _, _)
        return abs2(Ψ[2]) # = ⟨2|Ψ⟩⟨Ψ|2⟩
    end

    # We first optimize without the running cost, so that we can compare the
    # two results. We'll still pass `g_b`, but set `lambda_b = 0` to
    # effectively disable this. This (as opposed to just not passing `g_b`) is
    # something people are likely to do in real life, and gives as an
    # opportunity to test that code path.

    QuantumControl.set_default_ad_framework(Zygote; quiet = true)

    problem1 = ControlProblem(
        [trajectory],
        tlist;
        J_T = J_T_ss,
        iter_stop = 50,
        g_b,
        lambda_b = 0.0,
        check_convergence = res -> begin
            (res.J_T <= 1e-2) && "J_T < 10⁻²"
        end,
        prop_method = Cheby,
    );

    result1 = optimize(problem1; method = GRAPE)

    @test iszero(result1.J_b)
    @test iszero(result1.J_b_prev)

    using QuantumControl.Controls: substitute, get_controls
    H_opt1 = substitute(
        H,
        IdDict(ϵ => result1.optimized_controls[i] for (i, ϵ) in enumerate(get_controls(H)))
    );

    opt1_dynamics = propagate(ket1, H_opt1, tlist; method = Cheby, storage = true)
    Pmax1 = maximum(abs2.(opt1_dynamics[2, :]))
    @test Pmax1 > 0.5

    QuantumControl.set_default_ad_framework(nothing; quiet = true)

    function xi(Ψ, _, _, _)
        return ComplexF64[0, -Ψ[2], 0]
    end

    problem2 = ControlProblem(
        [trajectory],
        tlist;
        J_T = J_T_ss,
        iter_stop = 100,
        check_convergence = res -> begin
            (res.J_T <= 1e-2) && (res.J_b <= 1e-2)
        end,
        prop_method = Cheby,
        g_b,
        xi,
        lambda_b = 4e-1,
        print_iter_info = [
            "iter.",
            "J",
            "J_T",
            "J_b",
            "λ_b⋅J_b",
            "ǁ∇(J_T+λ_b·J_b)ǁ",
            "ǁΔϵǁ",
            "ΔJ",
        ],
        store_iter_info = ["J", "J_T", "J_b", "λ_b⋅J_b", "ǁ∇Jǁ"]
    );

    result2 = optimize(problem2; method = GRAPE)
    @test result2.iter > result1.iter + 10
    @test result2.converged
    @test result2.message == "Convergence check returned true"
    @test result2.J_b > 0.0
    @test result2.J_b_prev > 0.0

    H_opt2 = substitute(
        H,
        IdDict(ϵ => result2.optimized_controls[i] for (i, ϵ) in enumerate(get_controls(H)))
    );
    opt2_dynamics = propagate(ket1, H_opt2, tlist; method = Cheby, storage = true)
    Pmax2 = maximum(abs2.(opt2_dynamics[2, :]))

    @test Pmax2 / Pmax1 < 1e-1

    result3 = optimize(problem2; method = GRAPE, gradient_method = :taylor)
    @test result3.iter > result1.iter + 10
    @test result3.converged
    @test result3.message == "Convergence check returned true"
    @test result3.J_b > 0.0
    @test result3.J_b_prev > 0.0

    H_opt3 = substitute(
        H,
        IdDict(ϵ => result3.optimized_controls[i] for (i, ϵ) in enumerate(get_controls(H)))
    );
    opt3_dynamics = propagate(ket1, H_opt3, tlist; method = Cheby, storage = true)
    Pmax3 = maximum(abs2.(opt3_dynamics[2, :]))
    # Optimizations agree within 5% relative error
    @test (abs(Pmax3 - Pmax2) / Pmax3) < 0.05

    function xi_wrong(Ψ, _, _, _)
        return ComplexF64[0, Ψ[2], 0]  # incorrect sign
    end
    result4 = optimize(problem2; method = GRAPE, xi = xi_wrong)
    @test contains(result4.message, "ABNORMAL_TERMINATION_IN_LNSRCH")


end
