using Test
using StableRNGs: StableRNG
using LinearAlgebra

using QuantumControl.Controls: evaluate
using GRAPE: taylor_grad_step!
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector

@testset "taylor_grad_step" begin

    rng = StableRNG(3991576559)
    N = 10  # size of Hilbert space
    # We'll test with non-Hermitian Hamiltonians
    Ĥ₀ = random_matrix(N; rng)
    Ĥ₁ = random_matrix(N; rng)
    Ĥ₂ = random_matrix(N; rng)
    ϵ₁(t) = 1.0
    ϵ₂(t) = 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    Ĥ = evaluate(Ĥ_of_t, [0, 1], 1)
    Ψ = random_state_vector(N; rng)
    𝕚 = 1im
    dt = 1.25
    temp_states = Tuple(random_state_vector(N) for _ = 1:4)

    commutator(A, B) = A * B - B * A

    """Evaluate ∂/∂ϵ exp(-𝕚 Ĥ dt) via a Taylor operator expansion."""
    function U_grad(Ĥ, μ̂, dt)
        # See Eq. (14) in de Fouquieres et. al, JMR 212, 412 (2011)
        Û = exp(-𝕚 * Ĥ * dt)
        converged = false
        Ĉ = μ̂
        terms = [(-𝕚 * dt) * Ĉ]
        n = 2
        while !converged
            Ĉ = commutator(Ĥ, Ĉ)
            term = -((𝕚 * dt)^n / factorial(big(n))) * Ĉ
            push!(terms, term)
            converged = (norm(term) < 1e-16)
            n += 1
        end
        return Û * sum(terms)
    end

    # forward
    Φ̃₁ = U_grad(Ĥ, Ĥ₁, dt) * Ψ
    Ψ̃₁ = similar(Ψ)
    taylor_grad_step!(Ψ̃₁, Ψ, Ĥ, Ĥ₁, dt, temp_states)
    @test norm(Φ̃₁ - Ψ̃₁) < 1e-14
    Φ̃₂ = U_grad(Ĥ, Ĥ₂, dt) * Ψ
    Ψ̃₂ = similar(Ψ)
    taylor_grad_step!(Ψ̃₂, Ψ, Ĥ, Ĥ₂, dt, temp_states)
    @test norm(Φ̃₂ - Ψ̃₂) < 1e-14

    # backward
    dt = -dt
    Φ̃₁ = U_grad(Ĥ, Ĥ₁, dt) * Ψ
    Ψ̃₁ = similar(Ψ)
    taylor_grad_step!(Ψ̃₁, Ψ, Ĥ, Ĥ₁, dt, temp_states)
    @test norm(Φ̃₁ - Ψ̃₁) < 1e-14
    Φ̃₂ = U_grad(Ĥ, Ĥ₂, dt) * Ψ
    Ψ̃₂ = similar(Ψ)
    taylor_grad_step!(Ψ̃₂, Ψ, Ĥ, Ĥ₂, dt, temp_states)
    @test norm(Φ̃₂ - Ψ̃₂) < 1e-14

end
