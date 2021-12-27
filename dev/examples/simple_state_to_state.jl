using Printf
using QuantumControl
using LinearAlgebra
using Optim
using GRAPE # XXX
using QuantumControlBase: chain_infohooks
using GRAPELinesearchAnalysis
using LineSearches
using PyPlot: matplotlib
matplotlib.use("Agg")

using Test

ϵ(t) = 0.2 * QuantumControl.shapes.flattop(t, T = 5, t_rise = 0.3, func = :blackman);

"""Two-level-system Hamiltonian."""
function hamiltonian(Ω = 1.0, ϵ = ϵ)
    σ̂_z = ComplexF64[1 0; 0 -1]
    σ̂_x = ComplexF64[0 1; 1 0]
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return (Ĥ₀, (Ĥ₁, ϵ))
end;

H = hamiltonian();
@test length(H) == 2

tlist = collect(range(0, 5, length = 500));

function plot_control(pulse::Vector, tlist)
    fig, ax = matplotlib.pyplot.subplots(figsize = (6, 3))
    ax.plot(tlist, pulse)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    return fig
end

plot_control(ϵ::T, tlist) where {T<:Function} = plot_control([ϵ(t) for t in tlist], tlist)

function ket(label)
    result = Dict("0" => Vector{ComplexF64}([1, 0]), "1" => Vector{ComplexF64}([0, 1]))
    return result[string(label)]
end;

@test dot(ket(0), ket(1)) ≈ 0

objectives = [Objective(initial_state = ket(0), generator = H, target_state = ket(1))]

@test length(objectives) == 1

problem = ControlProblem(
    objectives = objectives,
    tlist = tlist,
    pulse_options=Dict(),
    iter_stop = 500,
    J_T = QuantumControl.functionals.J_T_sm,
    gradient=QuantumControl.functionals.grad_J_T_sm!,
    check_convergence = res -> begin
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
);

guess_dynamics = propagate_objective(
    objectives[1],
    problem.tlist;
    storage = true,
    observables = (Ψ -> abs.(Ψ) .^ 2,),
)

function plot_population(pop0::Vector, pop1::Vector, tlist)
    fig, ax = matplotlib.pyplot.subplots(figsize = (6, 3))
    ax.plot(tlist, pop0, label = "0")
    ax.plot(tlist, pop1, label = "1")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    return fig
end

println("")
opt_result = optimize_grape(
        problem,
        #=show_trace=true, extended_trace=false,=#
        info_hook=chain_infohooks(
            GRAPELinesearchAnalysis.plot_linesearch(@__DIR__),
            GRAPE.print_table,
        )
        #=alphaguess=LineSearches.InitialStatic(alpha=0.2),=#
        #=linesearch=LineSearches.HagerZhang(alphamax=2.0),=#
        #=linesearch=LineSearches.BackTracking(), # fails=#
        #=allow_f_increases=true,=#
);

opt_result

display(opt_result)
display(opt_result.optim_res)
@test opt_result.J_T < 1e-3

using UnicodePlots
println(lineplot(tlist, opt_result.optimized_controls[1]))

opt_dynamics = propagate_objective(
    objectives[1],
    problem.tlist;
    controls_map = IdDict(ϵ => opt_result.optimized_controls[1]),
    storage = true,
    observables = (Ψ -> abs.(Ψ) .^ 2,),
)

@test opt_dynamics[2,end] > 0.99

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

