# # Example 1: Optimization of a State-to-State Transfer in a Two-Level-System

#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`simple_state_to_state.ipynb`](@__NBVIEWER_ROOT_URL__/examples/simple_state_to_state.ipynb).
#md #
#md #     Compare this example against the [same example using Krotov's
#md #     method](https://juliaquantumcontrol.github.io/Krotov.jl/stable/examples/simple_state_to_state/).

#md # ``\gdef\op#1{\hat{#1}}``
#md # ``\gdef\init{\text{init}}``
#md # ``\gdef\tgt{\text{tgt}}``

#nb # $
#nb # \newcommand{tr}[0]{\operatorname{tr}}
#nb # \newcommand{diag}[0]{\operatorname{diag}}
#nb # \newcommand{abs}[0]{\operatorname{abs}}
#nb # \newcommand{pop}[0]{\operatorname{pop}}
#nb # \newcommand{aux}[0]{\text{aux}}
#nb # \newcommand{opt}[0]{\text{opt}}
#nb # \newcommand{tgt}[0]{\text{tgt}}
#nb # \newcommand{init}[0]{\text{init}}
#nb # \newcommand{lab}[0]{\text{lab}}
#nb # \newcommand{rwa}[0]{\text{rwa}}
#nb # \newcommand{bra}[1]{\langle#1\vert}
#nb # \newcommand{ket}[1]{\vert#1\rangle}
#nb # \newcommand{Bra}[1]{\left\langle#1\right\vert}
#nb # \newcommand{Ket}[1]{\left\vert#1\right\rangle}
#nb # \newcommand{Braket}[2]{\left\langle #1\vphantom{#2}\mid{#2}\vphantom{#1}\right\rangle}
#nb # \newcommand{op}[1]{\hat{#1}}
#nb # \newcommand{Op}[1]{\hat{#1}}
#nb # \newcommand{dd}[0]{\,\text{d}}
#nb # \newcommand{Liouville}[0]{\mathcal{L}}
#nb # \newcommand{DynMap}[0]{\mathcal{E}}
#nb # \newcommand{identity}[0]{\mathbf{1}}
#nb # \newcommand{Norm}[1]{\lVert#1\rVert}
#nb # \newcommand{Abs}[1]{\left\vert#1\right\vert}
#nb # \newcommand{avg}[1]{\langle#1\rangle}
#nb # \newcommand{Avg}[1]{\left\langle#1\right\rangle}
#nb # \newcommand{AbsSq}[1]{\left\vert#1\right\vert^2}
#nb # \newcommand{Re}[0]{\operatorname{Re}}
#nb # \newcommand{Im}[0]{\operatorname{Im}}
#nb # $


# This first example illustrates the basic use of the `GRAPE.jl` by solving a simple canonical optimization problem: the transfer of population in a two level system.

using DrWatson
@quickactivate "GRAPETests"
#-
using QuantumControl

#jl using Test
#jl println("")

# ## Two-level Hamiltonian

# We consider the Hamiltonian $\op{H}_{0} = - \frac{\omega}{2} \op{\sigma}_{z}$, representing a simple qubit with energy level splitting $\omega$ in the basis $\{\ket{0},\ket{1}\}$. The control field $\epsilon(t)$ is assumed to couple via the Hamiltonian $\op{H}_{1}(t) = \epsilon(t) \op{\sigma}_{x}$ to the qubit, i.e., the control field effectively drives transitions between both qubit states.
#
# We we will use

ϵ(t) = 0.2 * QuantumControl.Shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);

#-
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
#-

H = tls_hamiltonian();
#jl @test length(H.ops) == 2

# The control field here switches on from zero at $t=0$ to it's maximum amplitude
# 0.2 within the time period 0.3 (the switch-on shape is half a [Blackman pulse](https://en.wikipedia.org/wiki/Window_function#Blackman_window)).
# It switches off again in the time period 0.3 before the
# final time $T=5$). We use a time grid with 500 time steps between 0 and $T$:

tlist = collect(range(0, 5, length=500));

#-
using Plots
Plots.default(
    linewidth               = 3,
    size                    = (550, 300),
    legend                  = :right,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8)
)
#-
function plot_control(pulse::Vector, tlist)
    plot(tlist, pulse, xlabel="time", ylabel="amplitude", legend=false)
end

plot_control(ϵ::T, tlist) where {T<:Function} = plot_control([ϵ(t) for t in tlist], tlist);
#-
fig = plot_control(ϵ, tlist)
#jl display(fig)

# ## Optimization target

# First, we define a convenience function for the eigenstates.

function ket(label)
    result = Dict("0" => Vector{ComplexF64}([1, 0]), "1" => Vector{ComplexF64}([0, 1]))
    return result[string(label)]
end;

#-
#jl using LinearAlgebra
#jl @test dot(ket(0), ket(1)) ≈ 0
#-

# The physical objective of our optimization is to transform the initial state $\ket{0}$ into the target state $\ket{1}$ under the time evolution induced by the Hamiltonian $\op{H}(t)$.

objectives = [Objective(initial_state=ket(0), generator=H, target_state=ket(1))];

#-
#jl @test length(objectives) == 1
#-

# The full control problem includes this objective, information about the time grid for the dynamics, and the functional to be used (the square modulus of the overlap $\tau$ with the target state in this case).

using QuantumControl.Functionals: J_T_sm

problem = ControlProblem(
    objectives=objectives,
    tlist=tlist,
    pulse_options=Dict(),
    iter_stop=500,
    J_T=J_T_sm,
    check_convergence=res -> begin
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
);


# ## Simulate dynamics under the guess field

# Before running the optimization procedure, we first simulate the dynamics under the guess field $\epsilon_{0}(t)$. The following solves equation of motion for the defined objective, which contains the initial state $\ket{\Psi_{\init}}$ and the Hamiltonian $\op{H}(t)$ defining its evolution.

guess_dynamics = propagate_objective(
    objectives[1],
    problem.tlist;
    storage=true,
    observables=(Ψ -> abs.(Ψ) .^ 2,)
)

#-
function plot_population(pop0::Vector, pop1::Vector, tlist)
    fig = plot(tlist, pop0, label="0", xlabel="time", ylabel="population")
    plot!(fig, tlist, pop1; label="1")
end;
#-
fig = plot_population(guess_dynamics[1, :], guess_dynamics[2, :], tlist)
#jl display(fig)

# ## Optimization with LBFGSB

# In the following we optimize the guess field $\epsilon_{0}(t)$ such that the intended state-to-state transfer $\ket{\Psi_{\init}} \rightarrow \ket{\Psi_{\tgt}}$ is solved.

# The GRAPE package performs the optimization by calculating the gradient of $J_T$ with respect to the values of the control field at each point in time. This gradient is then fed into a backend solver that calculates an appropriate update based on that gradient.

# By default, this backend is [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl), a wrapper around the true and tested [L-BFGS-B Fortran library](http://users.iems.northwestern.edu/%7Enocedal/lbfgsb.html). L-BFGS-B is a pseudo-Hessian method: it efficiently estimates the second-order Hessian from the gradient information. The search direction determined from that Hessian dramatically improves convergence compared to using the gradient directly as a search direction. The L-BFGS-B method performs its own linesearch to determine how far to go in the search direction.

# It can be quite instructive to see how the improvement in the pseudo-Hessian search direction compares to the gradient, how the linesearch finds an appropriate step width. For this purpose, we have a [GRAPELinesearchAnalysis](https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl) package that automatically generates plots in every iteration of the optimization showing the linesearch behavior

using GRAPELinesearchAnalysis

# We feed this into the optimization as part of the `info_hook`.

opt_result_LBFGSB = @optimize_or_load(
    datadir("TLS", "opt_result_LBFGSB.jld2"),
    problem;
    method=:grape,
    force=true,
    info_hook=(
        GRAPELinesearchAnalysis.plot_linesearch(datadir("TLS", "Linesearch", "LBFGSB")),
        QuantumControl.GRAPE.print_table,
    )
);
#-
#jl @test opt_result_LBFGSB.J_T < 1e-3
#-

# When going through this tutorial locally, the [generated images for the linesearch](https://github.com/JuliaQuantumControl/GRAPE.jl/tree/data-dump/TLS/Linesearch/LBFGSB) can be found in `docs/TLS/Linesearch/LBFGSB`.

datadir("TLS", "Linesearch", "LBFGSB")
#-
opt_result_LBFGSB

# We can plot the optimized field:

#-
fig = plot_control(opt_result_LBFGSB.optimized_controls[1], tlist)
#jl display(fig)
#-


# ## Optimization via semi-automatic differentiation

# Our GRAPE implementation includes the analytic gradient of the optimization functional `J_T_sm`. Thus, we only had to pass the functional itself to the optimization. More generally, for functionals where the analytic gradient is not known, semi-automatic differentiation can be used to determine it automatically. For illustration, we may re-run the optimization forgoing the known analytic gradient and instead using an automatically determined gradient.

# As shown in Goerz et al., arXiv:2205.15044, by evaluating the gradient of ``J_T`` via a chain rule in the propagated states, the dependency of the gradient on the final time functional is pushed into the boundary condition for the backward propagation, ``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``. For functionals that can be written in terms of the overlaps ``τ_k`` of the forward-propagated states and target states, such as the `J_T_sm` used here, a further chain rule leaves derivatives of `J_T` with respect to the overlaps ``τ_k``, which are easily obtained via automatic differentiation. This happens automatically if we use `make_chi` with `force_zygote=true` and pass the resulting `chi` to the optimization:

using QuantumControl.Functionals: make_chi

chi_sm = make_chi(J_T_sm, objectives; force_zygote=true)

opt_result_LBFGSB_via_χ = optimize(problem; method=:grape, chi=chi_sm);
#-
opt_result_LBFGSB_via_χ

# ## Optimization with Optim.jl

# As an alternative to the default L-BFGS-B backend, we can also use any of the gradient-based optimizers in [Optiml.jl](https://github.com/JuliaNLSolvers/Optim.jl). This also gives full control over the linesearch method.

import Optim
import LineSearches

# Here, we use the LBFGS implementation that is part of Optim (which is not exactly the same as L-BFGS-B; "B" being the variant of LBFGS with optional additional bounds on the control) with a Hager-Zhang linesearch

opt_result_OptimLBFGS = @optimize_or_load(
    datadir("TLS", "opt_result_OptimLBFGS.jld2"),
    problem;
    method=:grape,
    force=true,
    info_hook=(
        GRAPELinesearchAnalysis.plot_linesearch(datadir("TLS", "Linesearch", "OptimLBFGS")),
        QuantumControl.GRAPE.print_table,
    ),
    optimizer=Optim.LBFGS(;
        alphaguess=LineSearches.InitialStatic(alpha=0.2),
        linesearch=LineSearches.HagerZhang(alphamax=2.0)
    )
);

#-
#jl @test opt_result_OptimLBFGS.J_T < 1e-3
#-

opt_result_OptimLBFGS
#-

# We can plot the optimized field:

fig = plot_control(opt_result_OptimLBFGS.optimized_controls[1], tlist)

# We can see that the choice of linesearch parameters in particular strongly influence the convergence and the resulting field. Play around with different methods and parameters, and compare the different [plots generated by `GRAPELinesearchAnalysis`](https://github.com/JuliaQuantumControl/GRAPE.jl/tree/data-dump/TLS/Linesearch/OptimLBFGS)!
#
# Empirically, we find the default L-BFGS-B to have a very well-behaved linesearch.

# ## Simulate the dynamics under the optimized field

# Having obtained the optimized control field, we can simulate the dynamics to verify that the optimized field indeed drives the initial state $\ket{\Psi_{\init}} = \ket{0}$ to the desired target state $\ket{\Psi_{\tgt}} = \ket{1}$.

using QuantumControl.Controls: substitute

opt_dynamics = propagate_objective(
    substitute(objectives[1], IdDict(ϵ => opt_result_LBFGSB.optimized_controls[1])),
    problem.tlist;
    storage=true,
    observables=(Ψ -> abs.(Ψ) .^ 2,)
)

#-
fig = plot_population(opt_dynamics[1, :], opt_dynamics[2, :], tlist)
#jl display(fig)
#-

#-
#jl @test opt_dynamics[2,end] > 0.99
#-
