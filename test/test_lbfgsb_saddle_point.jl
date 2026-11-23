# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test

using QuantumControl: hamiltonian, Trajectory, ControlProblem, optimize
using QuantumControl.Shapes: box
using QuantumControl.Amplitudes: ShapedAmplitude
using QuantumPropagators: Cheby
using QuantumControl.Functionals: J_T_sm
using LinearAlgebra: kron
using GRAPE

const ⊗ = kron


function two_qubit_hamiltonian(; ϵ1, ϵ2, ϵ3, ϵ4, ϵ5, ϵ6,)
    𝟙 = ComplexF64[
        1  0
        0  1
    ]
    σz = ComplexF64[
        1  0
        0 -1
    ]
    σx = ComplexF64[
        0  1
        1  0
    ]
    σy = ComplexF64[
        0  -1im
        1im  0
    ]
    Ĥ1 = σx ⊗ 𝟙
    Ĥ2 = σy ⊗ 𝟙
    Ĥ3 = σz ⊗ 𝟙
    Ĥ4 = 𝟙 ⊗ σx
    Ĥ5 = 𝟙 ⊗ σy
    Ĥ6 = 𝟙 ⊗ σz
    Ĥ0 = π / 2 * σy ⊗ σy
    return hamiltonian(
        Ĥ0,
        (Ĥ1, ϵ1),
        (Ĥ2, ϵ2),
        (Ĥ3, ϵ3),
        (Ĥ4, ϵ4),
        (Ĥ5, ϵ5),
        (Ĥ6, ϵ6)
    )
end;


function guess_amplitudes(; T = 1.0, E₀ = 0.1, dt = 0.001)

    tlist = collect(range(0, T, step = dt))
    shape(t) = box(t, 0.0, T)
    ϵ1 = ShapedAmplitude(t -> E₀, tlist; shape)
    ϵ2 = ShapedAmplitude(t -> E₀, tlist; shape)
    ϵ3 = ShapedAmplitude(t -> E₀, tlist; shape)
    ϵ4 = ShapedAmplitude(t -> E₀, tlist; shape)
    ϵ5 = ShapedAmplitude(t -> E₀, tlist; shape)
    ϵ6 = ShapedAmplitude(t -> E₀, tlist; shape)

    return tlist, ϵ1, ϵ2, ϵ3, ϵ4, ϵ5, ϵ6

end

function ket(i::Int64; N = 2)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(indices::Int64...; N = 2)
    Ψ = ket(indices[1]; N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N)
    end
    return Ψ
end

function ket(label::AbstractString; N = 2)
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N)
end;


@testset "CNOT with single-qubit drives and static interaction" begin

    tlist, Ω1, Ω2, Ω3, Ω4, Ω5, Ω6 = guess_amplitudes()
    CNOT = ComplexF64[
        1  0  0  0
        0  1  0  0
        0  0  0  1
        0  0  1  0
    ]
    basis = [ket("00"), ket("01"), ket("10"), ket("11")]
    basis_tgt = transpose(CNOT) * basis
    H = two_qubit_hamiltonian(ϵ1 = Ω1, ϵ2 = Ω2, ϵ3 = Ω3, ϵ4 = Ω4, ϵ5 = Ω5, ϵ6 = Ω6)
    trajectories = [
        Trajectory(initial_state = Ψ, target_state = Ψtgt, generator = H) for
        (Ψ, Ψtgt) ∈ zip(basis, basis_tgt)
    ]
    problem = ControlProblem(
        trajectories,
        tlist;
        iter_stop = 50,
        prop_method = Cheby,
        use_threads = true,
        J_T = J_T_sm,
    )

    # with "medium precision" (old defaults), this gets stuck at a saddle point
    opt_result = optimize(problem; method = GRAPE, lbfgsb_pgtol = 1e-5, lbfgsb_factr = 1e7)
    @test !opt_result.converged
    @test contains(opt_result.message, "NORM_OF_PROJECTED_GRADIENT_<=_PGTOL")
    @test abs(opt_result.J_T - 0.75) < 5e-3

    opt_result = optimize(problem; method = GRAPE)
    @test opt_result.converged
    @test opt_result.J_T < 1e-2

end
