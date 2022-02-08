using DrWatson
@quickactivate "GRAPETests"

using Printf
using QuantumControl
using LinearAlgebra
using GRAPELinesearchAnalysis
using LineSearches

using Plots
Plots.default(linewidth=3, size=(550, 300))

using Test

ϵ(t) = 0.2 * QuantumControl.Shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);

"""Two-level-system Hamiltonian."""
function hamiltonian(Ω=1.0, ϵ=ϵ)
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
    return (Ĥ₀, (Ĥ₁, ϵ))
end;

H = hamiltonian();
@test length(H) == 2

tlist = collect(range(0, 5, length=500));

function plot_control(pulse::Vector, tlist)
    plot(tlist, pulse, xlabel="time", ylabel="amplitude", legend=false)
end

plot_control(ϵ::T, tlist) where {T<:Function} = plot_control([ϵ(t) for t in tlist], tlist);

fig = plot_control(H[2][2], tlist)
display(fig)

function ket(label)
    result = Dict("0" => Vector{ComplexF64}([1, 0]), "1" => Vector{ComplexF64}([0, 1]))
    return result[string(label)]
end;

@test dot(ket(0), ket(1)) ≈ 0

objectives = [Objective(initial_state=ket(0), generator=H, target_state=ket(1))]

@test length(objectives) == 1

problem = ControlProblem(
    objectives=objectives,
    tlist=tlist,
    pulse_options=Dict(),
    iter_stop=500,
    J_T=QuantumControl.Functionals.J_T_sm,
    gradient=QuantumControl.Functionals.grad_J_T_sm!,
    check_convergence=res -> begin
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
);

guess_dynamics = propagate_objective(
    objectives[1],
    problem.tlist;
    storage=true,
    observables=(Ψ -> abs.(Ψ) .^ 2,)
)

function plot_population(pop0::Vector, pop1::Vector, tlist)
    legend_args = Dict(
        :legend => :right,
        :foreground_color_legend => nothing,
        :background_color_legend => RGBA(1, 1, 1, 0.8)
    )
    fig = plot(tlist, pop0, label="0", xlabel="time", ylabel="population")
    plot!(fig, tlist, pop1; label="1", legend_args...)
end;

fig = plot_population(guess_dynamics[1, :], guess_dynamics[2, :], tlist)
display(fig)

println("")
opt_result, file = @optimize_or_load(
    datadir(),
    problem,
    method = :grape,
    prefix = "TLSOCT",
    savename_kwargs = Dict(:ignores => ["chi"], :connector => "#"),
    #=show_trace=true, extended_trace=false,=#
    info_hook = chain_infohooks(
        GRAPELinesearchAnalysis.plot_linesearch(@__DIR__),
        QuantumControl.GRAPE.print_table,
    )
    #=alphaguess=LineSearches.InitialStatic(alpha=0.2),=#
    #=linesearch=LineSearches.HagerZhang(alphamax=2.0),=#
    #=linesearch=LineSearches.BackTracking(), # fails=#
    #=allow_f_increases=true,=#
);

opt_result

@test opt_result.J_T < 1e-3

fig = plot_control(opt_result.optimized_controls[1], tlist)
display(fig)

opt_dynamics = propagate_objective(
    objectives[1],
    problem.tlist;
    controls_map=IdDict(ϵ => opt_result.optimized_controls[1]),
    storage=true,
    observables=(Ψ -> abs.(Ψ) .^ 2,)
)

fig = plot_population(opt_dynamics[1, :], opt_dynamics[2, :], tlist)
display(fig)

@test opt_dynamics[2,end] > 0.99

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

