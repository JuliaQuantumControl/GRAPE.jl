using Test
using QuantumControl
using QuantumPropagators: ExpProp
using QuantumControl.Functionals: J_T_sm
using GRAPE
import Krotov
using LinearAlgebra
using Printf
import IOCapture
import Optim
using LineSearches
using StaticArrays: @SMatrix, @SVector
using GRAPE: step_width, pulse_update, search_direction, gradient, vec_angle

ϵ(t) = 0.2 * QuantumControl.Shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);


"""Two-level-system Hamiltonian."""
function tls_hamiltonian(Ω=1.0, ϵ=ϵ)
    σ̂_z = ComplexF64[
        1  0
        0 -1
    ]
    σ̂_x = ComplexF64[
        0  1
        1  0
    ]
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return hamiltonian(Ĥ₀, (Ĥ₁, ϵ))
end;


"""Two-level-system Hamiltonian, using StaticArrays."""
function tls_hamiltonian_static(Ω=1.0, ϵ=ϵ)
    σ̂_z = @SMatrix ComplexF64[
        1  0
        0 -1
    ]
    σ̂_x = @SMatrix ComplexF64[
        0  1
        1  0
    ]
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return hamiltonian(Ĥ₀, (Ĥ₁, ϵ))
end;


function ls_info_hook(wrk, iter)
    g = gradient(wrk)
    s = search_direction(wrk)
    Δu = pulse_update(wrk)
    if iter > 1
        @test abs(vec_angle(Δu, s)) < 1e-10
    end
    g_norm = norm(g)
    s_norm = norm(s)
    ratio = s_norm / g_norm
    # angle is between the negative gradient and the search direction, in
    # degrees
    angle = vec_angle(-g, s; unit=:degree)
    α = step_width(wrk)
    if iter > 1
        @test norm(Δu - α * s) < 1e-10
    end
    return (iter, g_norm, s_norm, ratio, angle, α)
end


function print_ls_table(res)
    println("")
    @printf("%6s", "iter")
    @printf("%10s", "|grad|")
    @printf("%10s", "|search|")
    @printf("%10s", "ratio")
    @printf("%10s", "angle(°)")
    @printf("%10s", "step α")
    println("")
    for (iter, g_norm, s_norm, ratio, angle, α) in res.records
        @printf("%6d", iter)
        @printf("%10.2e", g_norm)
        @printf("%10.2e", s_norm)
        @printf("%10.2f", ratio)
        @printf("%10.2f", angle)
        @printf("%10.2f", α)
        println("")
    end
    println("")
end


@testset "TLS (LBFGS.jl)" begin

    println("\n================== TLS (LBFGS.jl) ==================\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
        callback=ls_info_hook,
    )
    res = optimize(problem; method=GRAPE)
    print_ls_table(res)
    display(res)
    @test res.J_T < 1e-3
    @test 0.75 < maximum(abs.(res.optimized_controls[1])) < 0.85
    println("===================================================\n")

end


@testset "TLS (LBFGS.jl-bound)" begin

    println("\n=============== TLS (LBFGS.jl-bound) ===============\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        upper_bound=0.7,
        lower_bound=-0.7,
        iter_stop=10,
        prop_method=ExpProp,
        J_T=J_T_sm,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
        callback=ls_info_hook,
    )
    res = optimize(problem; method=GRAPE)
    print_ls_table(res)
    display(res)
    @test res.J_T < 1e-3
    @test 0.65 < maximum(abs.(res.optimized_controls[1])) < 0.700001
    println("===================================================\n")

end


@testset "TLS (LBFGS.jl) trace debugging" begin

    captured = IOCapture.capture(passthrough=false) do
        println("\n================== TLS (LBFGS.jl) ==================\n")
        H = tls_hamiltonian()
        tlist = collect(range(0, 5, length=501))
        Ψ₀ = ComplexF64[1, 0]
        Ψtgt = ComplexF64[0, 1]
        problem = ControlProblem(
            [Trajectory(Ψ₀, H, target_state=Ψtgt)],
            tlist;
            iter_stop=5,
            prop_method=ExpProp,
            J_T=J_T_sm,
            callback=ls_info_hook,
            lbfgsb_iprint=100,
        )
        res = optimize(problem; method=GRAPE)
        print_ls_table(res)
        display(res)
        println("===================================================\n")
        res
    end
    #write("debug.log", captured.output)
    @test contains(captured.output, "RUNNING THE L-BFGS-B CODE")
    res = captured.value
    @test res.J_T < 1e-3
    @test 0.75 < maximum(abs.(res.optimized_controls[1])) < 0.85

end


@testset "TLS (LBFGS.jl-Taylor)" begin

    println("\n============== TLS (LBFGS.jl-Taylor) ==============\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        gradient_method=:taylor,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
        callback=ls_info_hook,
    )
    res = optimize(problem; method=GRAPE)
    print_ls_table(res)
    display(res)
    @test res.J_T < 1e-3
    @test 0.75 < maximum(abs.(res.optimized_controls[1])) < 0.85
    println("===================================================\n")

end


@testset "TLS (Optim.jl-LBFGS-HZ)" begin

    println("\n============= TLS (Optim.jl-LBFGS-HZ) =============\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        optimizer=Optim.LBFGS(;
            alphaguess=LineSearches.InitialStatic(alpha=0.2),
            linesearch=LineSearches.HagerZhang(alphamax=100.0)
        ),
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
        callback=ls_info_hook,
    )
    res = optimize(problem; method=GRAPE)
    print_ls_table(res)
    display(res)
    @test res.J_T < 1e-3
    @test 0.75 < maximum(abs.(res.optimized_controls[1])) < 0.85
    println("===================================================\n")

end


@testset "TLS (static)" begin

    println("\n================ TLS (static) ======================\n")
    H = tls_hamiltonian_static()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = @SVector ComplexF64[1, 0]
    Ψtgt = @SVector ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        rethrow_exceptions=true,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
    )
    res = optimize(problem; method=GRAPE)
    display(res)
    @test res.J_T < 1e-3
    @test 0.75 < maximum(abs.(res.optimized_controls[1])) < 0.85
    println("===================================================\n")

end


@testset "TLS (continue from Krotov)" begin

    println("\n============ TLS (Krotov continuation) ============\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
    )
    res_krotov = optimize(problem; method=Krotov, lambda_a=100.0, iter_stop=2)
    @test res_krotov.iter == 2
    res =
        optimize(problem; method=GRAPE, continue_from=res_krotov, store_iter_info=["J_T"],)
    display(res)
    @test res.J_T < 2e-2
    @test length(res.records) == 4
    @test abs(res.records[1][1] - res_krotov.J_T) < 1e-14
    println("===================================================\n")

end


@testset "TLS (continue with Krotov)" begin

    println("\n=========== TLS (continue with Krotov) ============\n")
    H = tls_hamiltonian()
    tlist = collect(range(0, 5, length=501))
    Ψ₀ = ComplexF64[1, 0]
    Ψtgt = ComplexF64[0, 1]
    problem = ControlProblem(
        [Trajectory(Ψ₀, H, target_state=Ψtgt)],
        tlist;
        iter_stop=5,
        prop_method=ExpProp,
        J_T=J_T_sm,
        check_convergence=res -> begin
            ((res.J_T < 1e-10) && (res.converged = true) && (res.message = "J_T < 10⁻¹⁰"))
        end,
    )
    res_grape = optimize(problem; method=GRAPE, iter_stop=2)
    res = optimize(
        problem;
        method=Krotov,
        continue_from=res_grape,
        lambda_a=1.0,
        store_iter_info=["J_T"],
    )
    @test length(res.records) == 4
    display(res)
    @test res.J_T < 1e-3
    @test abs(res.records[1][1] - res_grape.J_T) < 1e-14
    println("===================================================\n")

end
