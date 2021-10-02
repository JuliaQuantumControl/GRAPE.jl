# # Example 1: Optimization of a State-to-State Transfer in a Two-Level-System

#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`simple_state_to_state.ipynb`](@__NBVIEWER_ROOT_URL__/examples/simple_state_to_state.ipynb).
#md #
#md #     Compare this example against the [same example using the `krotov`
#md #     Python package](https://qucontrol.github.io/krotov/v1.2.1/notebooks/01_example_simple_state_to_state.html).

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


# This first example illustrates the basic use of the `Krotov.jl` by solving a
# simple canonical optimization problem: the transfer of population in a two
# level system.

using QuantumControl
using LinearAlgebra

#jl using Test

# ## Two-level Hamiltonian

# We consider the Hamiltonian $\op{H}_{0} = - \frac{\omega}{2} \op{\sigma}_{z}$, representing
# a simple qubit with energy level splitting $\omega$ in the basis
# $\{\ket{0},\ket{1}\}$. The control field $\epsilon(t)$ is assumed to couple via
# the Hamiltonian $\op{H}_{1}(t) = \epsilon(t) \op{\sigma}_{x}$ to the qubit,
# i.e., the control field effectively drives transitions between both qubit
# states.
#
# We we will use

ϵ(t) = 0.2 * QuantumControl.shapes.flattop(t, T=5, t_rise=0.3, func=:blackman);


#-
"""Two-level-system Hamiltonian."""
function hamiltonian(Ω=1.0, ϵ=ϵ)
    σ̂_z = ComplexF64[1 0; 0 -1];
    σ̂_x = ComplexF64[0 1; 1  0];
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return (Ĥ₀, (Ĥ₁, ϵ))
end
;
#-

H = hamiltonian();
#jl @test length(H) == 2

# The control field here switches on from zero at $t=0$ to it's maximum amplitude
# 0.2 within the time period 0.3 (the switch-on shape is half a [Blackman pulse](https://en.wikipedia.org/wiki/Window_function#Blackman_window)).
# It switches off again in the time period 0.3 before the
# final time $T=5$). We use a time grid with 500 time steps between 0 and $T$:

tlist = collect(range(0, 5, length=500));

#-

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

#!jl plot_control(H[2][2], tlist)

# ## Optimization target

# The `krotov` package requires the goal of the optimization to be described by a
# list of `Objective` instances. In this example, there is only a single
# objective: the state-to-state transfer from initial state $\ket{\Psi_{\init}} =
# \ket{0}$ to the target state $\ket{\Psi_{\tgt}} = \ket{1}$, under the dynamics
# of the Hamiltonian $\op{H}(t)$:

function ket(label)
    result = Dict(
        "0" => Vector{ComplexF64}([1, 0]),
        "1" => Vector{ComplexF64}([0, 1]),
    )
    return result[string(label)]
end
;

#-
#jl @test dot(ket(0), ket(1)) ≈ 0
#-

objectives = [
    Objective(initial_state=ket(0), generator=H, target_state=ket(1))
]

#-
#jl @test length(objectives) == 1
#-

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

# ## Simulate dynamics under the guess field

# Before running the optimization procedure, we first simulate the dynamics under the
# guess field $\epsilon_{0}(t)$. The following solves equation of motion for the
# defined objective, which contains the initial state $\ket{\Psi_{\init}}$ and
# the Hamiltonian $\op{H}(t)$ defining its evolution.

guess_dynamics = propagate(
        objectives[1], problem.tlist;
        storage=true, observables=(Ψ->abs.(Ψ).^2, )
)

#-

function plot_population(pop0::Vector, pop1::Vector, tlist)
    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 3))
    ax.plot(tlist, pop0, label="0")
    ax.plot(tlist, pop1, label="1")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    return fig
end

#!jl plot_population(guess_dynamics[1,:], guess_dynamics[2,:], tlist)

# ## Optimize

# In the following we optimize the guess field $\epsilon_{0}(t)$ such
# that the intended state-to-state transfer $\ket{\Psi_{\init}} \rightarrow
# \ket{\Psi_{\tgt}}$ is solved, via the `krotov` package's central
# `optimize_pulses` routine.  It requires, besides the previously defined
# `objectives`, information about the optimization functional $J_T$ (implicitly,
# via `chi_constructor`, which calculates the states $\ket{\chi} =
# \frac{J_T}{\bra{\Psi}}$).

opt_result = optimize(problem, method=:krotov);
#-
opt_result
#-
#jl @test opt_result.J_T < 1e-3
#-

# We can plot the optimized field:

#-
#!jl plot_control(opt_result.optimized_controls[1], tlist)
#-

# ## Simulate the dynamics under the optimized field

# Having obtained the optimized control field, we can simulate the dynamics to
# verify that the optimized field indeed drives the initial state
# $\ket{\Psi_{\init}} = \ket{0}$ to the desired target state
# $\ket{\Psi_{\tgt}} = \ket{1}$.

opt_dynamics = propagate(
        objectives[1], problem.tlist;
        controls_map=IdDict(ϵ  => opt_result.optimized_controls[1]),
        storage=true, observables=(Ψ->abs.(Ψ).^2, )
)

#-
#!jl plot_population(opt_dynamics[1,:], opt_dynamics[2,:], tlist)
#-

#-
#jl @test opt_dynamics[2,end] > 0.99
#-
