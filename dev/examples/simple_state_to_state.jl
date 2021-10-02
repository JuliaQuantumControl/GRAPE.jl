using QuantumControl
using LinearAlgebra

using Test

ϵ(t) = 0.2 * QuantumControl.shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);

"""Two-level-system Hamiltonian."""
function hamiltonian(Ω=1.0, ϵ=ϵ)
    σ̂_z = ComplexF64[1 0; 0 -1];
    σ̂_x = ComplexF64[0 1; 1  0];
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return (Ĥ₀, (Ĥ₁, ϵ))
end
;

H = hamiltonian();
@test length(H) == 2

tlist = collect(range(0, 5, length=500));

using PyPlot
matplotlib.use("Agg")

function plot_control(pulse::Vector, tlist)
    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 3))
    ax.plot(tlist, pulse)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    return fig
end

plot_control(ϵ::T, tlist) where T<:Function =
    plot_control([ϵ(t) for t in tlist], tlist)

function ket(label)
    result = Dict(
        "0" => Vector{ComplexF64}([1, 0]),
        "1" => Vector{ComplexF64}([0, 1]),
    )
    return result[string(label)]
end
;

@test dot(ket(0), ket(1)) ≈ 0

objectives = [
    Objective(initial_state=ket(0), generator=H, target_state=ket(1))
]

@test length(objectives) == 1

problem = ControlProblem(
    objectives=objectives,
    pulse_options=IdDict(
        ϵ  => Dict(
            :lambda_a => 5,
            :update_shape => t -> QuantumControl.shapes.flattop(
                t, T=5, t_rise=0.3, func=:blackman
            ),
        )
    ),
    tlist=tlist,
    iter_stop=50,
    chi=QuantumControl.functionals.chi_ss!,
    J_T=QuantumControl.functionals.J_T_ss,
    check_convergence= res -> begin (
            (res.J_T < 1e-3)
            && (res.converged = true)
            && (res.message="J_T < 10⁻³")
        ) end
);

guess_dynamics = propagate(
        objectives[1], problem.tlist;
        storage=true, observables=(Ψ->abs.(Ψ).^2, )
)

function plot_population(pop0::Vector, pop1::Vector, tlist)
    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 3))
    ax.plot(tlist, pop0, label="0")
    ax.plot(tlist, pop1, label="1")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    return fig
end

opt_result = optimize(problem, method=:krotov);

opt_result

@test opt_result.J_T < 1e-3

opt_dynamics = propagate(
        objectives[1], problem.tlist;
        controls_map=IdDict(ϵ  => opt_result.optimized_controls[1]),
        storage=true, observables=(Ψ->abs.(Ψ).^2, )
)

@test opt_dynamics[2,end] > 0.99

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

