# # Example 2: Entangling quantum gates for coupled transmon qubits

#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`perfect_entanglers.ipynb`](@__NBVIEWER_ROOT_URL__/examples/perfect_entanglers.ipynb).
#md #
#md #     Compare this example against the [related example using Krotov's
#md #     method](https://juliaquantumcontrol.github.io/Krotov.jl/stable/examples/perfect_entanglers/).

#md # ``\gdef\Op#1{\hat{#1}}``
#md # ``\gdef\op#1{\hat{#1}}``
#md # ``\gdef\init{\text{init}}``
#md # ``\gdef\tgt{\text{tgt}}``
#md # ``\gdef\Re{\operatorname{Re}}``
#md # ``\gdef\Im{\operatorname{Im}}``

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

using DrWatson
@quickactivate "GRAPETests"
#jl using Test; println("")

# This example illustrates the optimization towards a perfectly entangling
# two-qubit gate for a system of two transmon qubits with a shared transmission
# line. It goes through three progressively more advanced optimizations:
#
# 1. The direct optimization for a ``\Op{O} = \sqrt{\text{iSWAP}}`` gate with a
#    standard square-modulus functional
# 2. The optimization towards a perfect entangler using the functional
#    developed in Goerz *et al.*, Phys. Rev. A 91, 062307
#    (2015) [GoerzPRA2015](@cite)
# 3. The direct maximization of of the gate concurrence
#
# While the first example evaluates the gradient of the optimization functional
# analytically, the latter two are examples for the use of automatic
# differentiation, or more specifically semi-automatic differentiation, as
# developed in [GoerzQ2022](@citet). The optimization of the gate
# concurrence specifically illustrates the optimization of a functional that is
# inherently non-analytical.

# ## Hamiltonian and guess pulses

# We will write the Hamiltonian in units of GHz (angular frequency; the factor
# 2π is implicit) and ns:

const GHz = 2π
const MHz = 0.001GHz
const ns = 1.0
const μs = 1000ns;

# The Hamiltonian and parameters are taken from
# Ref. [GoerzPRA2015; Table 1](@cite).

⊗ = kron
const 𝕚 = 1im
const N = 6  # levels per transmon

using LinearAlgebra
using SparseArrays
using QuantumControl


function transmon_hamiltonian(;
    Ωre,
    Ωim,
    N=N,  # levels per transmon
    ω₁=4.380GHz,
    ω₂=4.614GHz,
    ωd=4.498GHz,
    α₁=-210MHz,
    α₂=-215MHz,
    J=-3MHz,
    λ=1.03,
    use_sparse=:auto
)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, N, N))
    b̂₁ = spdiagm(1 => complex.(sqrt.(collect(1:N-1)))) ⊗ 𝟙
    b̂₂ = 𝟙 ⊗ spdiagm(1 => complex.(sqrt.(collect(1:N-1))))
    b̂₁⁺ = sparse(b̂₁')
    b̂₂⁺ = sparse(b̂₂')
    n̂₁ = sparse(b̂₁' * b̂₁)
    n̂₂ = sparse(b̂₂' * b̂₂)
    n̂₁² = sparse(n̂₁ * n̂₁)
    n̂₂² = sparse(n̂₂ * n̂₂)
    b̂₁⁺_b̂₂ = sparse(b̂₁' * b̂₂)
    b̂₁_b̂₂⁺ = sparse(b̂₁ * b̂₂')

    ω̃₁ = ω₁ - ωd
    ω̃₂ = ω₂ - ωd

    Ĥ₀ = sparse(
        (ω̃₁ - α₁ / 2) * n̂₁ +
        (α₁ / 2) * n̂₁² +
        (ω̃₂ - α₂ / 2) * n̂₂ +
        (α₂ / 2) * n̂₂² +
        J * (b̂₁⁺_b̂₂ + b̂₁_b̂₂⁺)
    )

    Ĥ₁re = (1 / 2) * (b̂₁ + b̂₁⁺ + λ * b̂₂ + λ * b̂₂⁺)
    Ĥ₁im = (𝕚 / 2) * (b̂₁⁺ - b̂₁ + λ * b̂₂⁺ - λ * b̂₂)

    if ((N < 5) && (use_sparse ≢ true)) || use_sparse ≡ false
        H = hamiltonian(Array(Ĥ₀), (Array(Ĥ₁re), Ωre), (Array(Ĥ₁im), Ωim))
    else
        H = hamiltonian(Ĥ₀, (Ĥ₁re, Ωre), (Ĥ₁im, Ωim))
    end
    return H

end;

# We choose a pulse duration of 400 ns. The guess pulse amplitude is 35 MHz,
# with a 15 ns switch-on/-off time. This switch-on/-off must be maintained in
# the optimization: A pulse that does not start from or end at zero would not
# be physical. For GRAPE, we can achieve this by using a `ShapedAmplitude`:

using QuantumControl.Amplitudes: ShapedAmplitude

# This allows to have a control amplitude ``Ω(t) = S(t) ϵ(t)`` where ``S(t)``
# is a fixed shape and ``ϵ(t)`` is the pulse directly tuned by the
# optimization. We start with a constant ``ϵ(t)`` and do not place any
# restrictions on how the optimization might update ``ϵ(t)``.

# The Hamiltonian is written in a rotating frame, so in general, the control
# field is allowed to be complex-valued. We separate this into two control
# fields, one for the real part and one for the imaginary part. Initially, the
# imaginary part is zero, corresponding to a field exactly at the frequency of
# the rotating frame.

# Note that passing `tlist` to `ShapedAmplitude` discretizes both the control
# and the shape function to the midpoints of the `tlist` array.

using QuantumControl.Shapes: flattop

function guess_amplitudes(; T=400ns, E₀=35MHz, dt=0.1ns, t_rise=15ns)

    tlist = collect(range(0, T, step=dt))
    shape(t) = flattop(t, T=T, t_rise=t_rise)
    Ωre = ShapedAmplitude(t -> E₀, tlist; shape)
    Ωim = ShapedAmplitude(t -> 0.0, tlist; shape)

    return tlist, Ωre, Ωim

end

tlist, Ωre_guess, Ωim_guess = guess_amplitudes();

#-

# We can visualize this:

using Plots
Plots.default(
    linewidth               = 3,
    size                    = (550, 300),
    legend                  = :right,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8),
)
using QuantumControl.Controls: discretize

function plot_complex_pulse(tlist, Ω; time_unit=:ns, ampl_unit=:MHz, kwargs...)

    Ω = discretize(Ω, tlist)  # make sure Ω is defined on *points* of `tlist`

    ax1 = plot(
        tlist ./ eval(time_unit),
        abs.(Ω) ./ eval(ampl_unit);
        label="|Ω|",
        xlabel="time ($time_unit)",
        ylabel="amplitude ($ampl_unit)",
        kwargs...
    )

    ax2 = plot(
        tlist ./ eval(time_unit),
        angle.(Ω) ./ π;
        label="ϕ(Ω)",
        xlabel="time ($time_unit)",
        ylabel="phase (π)"
    )

    plot(ax1, ax2, layout=(2, 1))

end

plot_complex_pulse(tlist, Array(Ωre_guess) .+ 𝕚 .* Array(Ωim_guess))

# We now instantiate the Hamiltonian with these control fields:

H = transmon_hamiltonian(Ωre=Ωre_guess, Ωim=Ωim_guess)

# ## Logical basis for two-qubit gates

# For simplicity, we will be define the qubits in the *bare* basis, i.e.
# ignoring the static coupling $J$.

function ket(i::Int64; N=N)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(indices::Int64...; N=N)
    Ψ = ket(indices[1]; N=N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N=N)
    end
    return Ψ
end

function ket(label::AbstractString; N=N)
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N=N)
end;

#-

basis = [ket("00"), ket("01"), ket("10"), ket("11")];

# ## Optimizing for a specific quantum gate

# Our target gate is ``\Op{O} = \sqrt{\text{iSWAP}}``:

SQRTISWAP = [
    1  0    0   0
    0 1/√2 𝕚/√2 0
    0 𝕚/√2 1/√2 0
    0  0    0   1
];

# For each basis state, we get a target state that results from applying the
# gate to the basis state (you can convince yourself that this equivalent
# multiplying the transpose of the above gate matrix to the vector of basis
# states):

basis_tgt = transpose(SQRTISWAP) * basis;

# The mapping from each initial (basis) state to the corresponding target state
# constitutes an "objective" for the optimization:

#-
objectives = [
    Objective(initial_state=Ψ, target_state=Ψtgt, generator=H) for
    (Ψ, Ψtgt) ∈ zip(basis, basis_tgt)
];

# We can analyze how all of the basis states evolve under the guess controls in
# one go:

guess_states = propagate_objectives(objectives, tlist; use_threads=true);

# The gate implemented by the guess controls is

U_guess = [basis[i] ⋅ guess_states[j] for i = 1:4, j = 1:4];

# We will optimize these objectives with a square-modulus functional

using QuantumControl.Functionals: J_T_sm

# The initial value of the functional is

J_T_sm(guess_states, objectives)

# which is the gate error

1 - (abs(tr(U_guess' * SQRTISWAP)) / 4)^2

# Now, we define the full optimization problems on top of the list of
# objectives, and with the optimization functional:

problem = ControlProblem(
    objectives=objectives,
    tlist=tlist,
    iter_stop=100,
    J_T=J_T_sm,
    check_convergence=res -> begin
        (
            (res.J_T > res.J_T_prev) &&
            (res.converged = true) &&
            (res.message = "Loss of monotonic convergence")
        )
        ((res.J_T <= 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
    use_threads=true,
);

#-

opt_result = @optimize_or_load(datadir("GATE_OCT.jld2"), problem; method=:GRAPE);
#-
opt_result

# We extract the optimized control field from the optimization result and plot
# the resulting amplitude.

# The `optimized_controls` field of the `opt_results` contains the optimized
# controls ``ϵ(t)``.

ϵ_opt = opt_result.optimized_controls[1] + 𝕚 * opt_result.optimized_controls[2];

# These must still be multiplied by the static shape ``S(t)`` that we set up
# for the guess amplitudes

Ω_opt = ϵ_opt .* discretize(Ωre_guess.shape, tlist)

plot_complex_pulse(tlist, Ω_opt)

# We then propagate the optimized control field to analyze the resulting
# quantum gate:

using QuantumControl.Controls: get_controls, substitute

opt_states = propagate_objectives(
    substitute(
        objectives,
        IdDict(zip(get_controls(objectives), opt_result.optimized_controls))
    ),
    tlist;
    use_threads=true
);


# The resulting gate is

U_opt = [basis[i] ⋅ opt_states[j] for i = 1:4, j = 1:4];

# and we can verify the resulting fidelity

(abs(tr(U_opt' * SQRTISWAP)) / 4)^2

# ## Optimizing for a general perfect entangler

# We define the optimization with one objective for each of the four basis
# states:

objectives = [Objective(; initial_state=Ψ, generator=H) for Ψ ∈ basis];

# Note that we omit the `target_state` here. This is because we will be
# optimizing for an arbitrary perfect entangler, not for a specific quantum
# gate. Thus, there is no a-priori known target state to which the initial
# state must evolve.

# The optimization is steered by the perfect entanglers distance measure
# $D_{PE}$, that is, the geometric distance of the quantum gate obtained from
# propagating the four basis states to the polyhedron of perfect entanglers in
# the Weyl chamber. Since the logical subspace defining the qubit is embedded
# in the larger Hilbert space of the transmon, there may be loss of population
# from the logical subspace. To counter this possibility in the optimization,
# we add a unitarity measure  to $D_{PE}$. The two terms are added with equal
# weight.

using TwoQubitWeylChamber: D_PE, gate_concurrence, unitarity
using QuantumControl.Functionals: gate_functional

J_T_PE = gate_functional(D_PE; unitarity_weight=0.5);

# The `gate_functional` routines used above converts the function `D_PE` that
# receives the gate $Û$ as a 4×4 matrix into a functional of the correct from
# for the `QuantumControl.optimize` routine, which is a function of the
# propagated states.

# We can check that for the guess pulse, we are not implementing a perfect
# entangler

gate_concurrence(U_guess)

#jl @test gate_concurrence(U_guess) < 0.9

# We find that the guess pulse produces a gate in the `W0*` region of the Weyl
# chamber:

using TwoQubitWeylChamber: weyl_chamber_region
weyl_chamber_region(U_guess)

#jl @test weyl_chamber_region(U_guess) == "W0*"

# That is, the region of the Weyl chamber containing controlled-phase gates with
# a phase $> π$ (Weyl chamber coordinates $c₁ > π/2$, $c₂ < π/4$).

# This in fact allows use to use the perfect entangler functional without
# modification: if the guess pulse were in the "W1" region of the Weyl chamber,
# (close to SWAP), we would have to flip its sign, or we would optimize towards
# the local equivalence class of the SWAP gate instead of towards the perfect
# of perfect entanglers. In principle, we could use a modified functional that
# takes the absolute square of the `D_PE` term, by using
#
# ```
# J_T_PE = gate_functional(D_PE; unitarity_weight=0.5, absolute_square=true)
# ```
#
# This would specifically optimize for the *surface* of the perfect
# entanglers functional.

# The guess pulse loses about 10% of population from the logical subspace:

1 - unitarity(U_guess)

#jl @test round(1 - unitarity(U_guess), digits=1) ≈ 0.1

# We can also evaluate the geometric distance to the polyhedron of perfect
# entanglers in the Weyl chamber:

D_PE(U_guess)

# Together with the unitarity measure, this is the initial value of the
# optimization functional:

0.5 * D_PE(U_guess) + 0.5 * (1 - unitarity(U_guess))
#-
J_T_PE(guess_states, objectives)

#jl @test 0.4 < J_T_PE(guess_states, objectives) < 0.5
#jl @test 0.5 * D_PE(U_guess) + 0.5 * (1-unitarity(U_guess)) ≈ J_T_PE(guess_states, objectives) atol=1e-15


# For the standard functional `J_T_sm` used in the previous section, our GRAPE
# was able to automatically use an analytic implementation of the gradient. For
# the perfect-entanglers functional, an analytic gradient exist, but is very
# cumbersome to implement. Instead, we make use of semi-automatic
# differentiation. As shown in Goerz et al., arXiv:2205.15044, by evaluating
# the gradient via a chain rule in the propagated states, the dependency of the
# gradient on the final time functional is pushed into the boundary condition
# for the backward propagation, ``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``. We can further
# exploit that `J_T` is an explicit function of the two-qubit gate in the
# computational basis and use a chain rule with respect to the elements of the
# two-qubit gate ``U_{kk'}``. The remaining derivatives ``∂J_T/∂U_{kk'}`` are
# then obtained via automatic differentiation. This is set up via the
# `make_gate_chi` function,

using QuantumControl.Functionals: make_gate_chi
chi_pe = make_gate_chi(D_PE, objectives; unitarity_weight=0.5);

# where the resulting `chi_pe` must be passed to the optimization.

# Now, we formulate the full control problem

problem = ControlProblem(
    objectives=objectives,
    tlist=tlist,
    iter_stop=100,
    J_T=J_T_PE,
    chi=chi_pe,
    check_convergence=res -> begin
        (
            (res.J_T > res.J_T_prev) &&
            (res.converged = true) &&
            (res.message = "Loss of monotonic convergence")
        )
        (
            (res.J_T <= 1e-3) &&
            (res.converged = true) &&
            (res.message = "Found a perfect entangler")
        )
    end,
    use_threads=true,
);

# With this, we can easily find a solution to the control problem:

opt_result = @optimize_or_load(datadir("PE_OCT.jld2"), problem; method=:GRAPE);
#-
opt_result


# We extract the optimized control field from the optimization result and plot
# it

ϵ_opt = opt_result.optimized_controls[1] + 𝕚 * opt_result.optimized_controls[2]
Ω_opt = ϵ_opt .* discretize(Ωre_guess.shape, tlist)

plot_complex_pulse(tlist, Ω_opt)

# We then propagate the optimized control field to analyze the resulting
# quantum gate:

opt_states = propagate_objectives(
    substitute(
        objectives,
        IdDict(zip(get_controls(objectives), opt_result.optimized_controls))
    ),
    tlist;
    use_threads=true
);

U_opt = [basis[i] ⋅ opt_states[j] for i = 1:4, j = 1:4];

# We find that we have achieved a perfect entangler:

gate_concurrence(U_opt)
#jl @test round(gate_concurrence(U_opt), digits=3) ≈ 1.0

# Moreover, we have reduced the population loss to less than 4%

1 - unitarity(U_opt)

#jl @test 1 - unitarity(U_opt) < 0.04


# ## Direct maximization of the gate concurrence

# In the previous optimizations, we have optimized for a perfect entangler
# indirectly via a geometric function in the Weyl chamber. The entire reason
# that perfect entangler functional was formulated is because calculating the
# gate concurrence directly involves the eigenvalues of the unitary, see
# [KrausPRA2001](@citet) and [ChildsPRA2003](@citet), which are inherently
# non-analytic.

# However, since we are able to obtain gradient from automatic differentiation,
# this is no longer an insurmountable obstacle

# We can define a functional for a given gate `U` that combines the gate
# concurrence and (as above) a unitarity measure to penalize loss of population
# from the logical subspace:

J_T_C = U -> 0.5 * (1 - gate_concurrence(U)) + 0.5 * (1 - unitarity(U));

# In the optimization, we will convert this functional to one that takes the
# propagated states as arguments (via the `gate_functional` routine).
# Also, as before, we have to create a matching routine for the boundary condition
# ``|χ_k⟩ = -\frac{∂}{∂⟨ϕ_k|} J_T`` of the backward-propagation via the
# `make_gate_chi` routine.

# Running this, we again are able to find a perfect entangler.

opt_result_direct = @optimize_or_load(
    datadir("PE_OCT_direct.jld2"),
    problem;
    method=:GRAPE,
    J_T=gate_functional(J_T_C),
    chi=make_gate_chi(J_T_C, objectives)
);
#-
opt_result_direct
#-
opt_states_direct = propagate_objectives(
    substitute(
        objectives,
        IdDict(zip(get_controls(objectives), opt_result_direct.optimized_controls))
    ),
    tlist;
    use_threads=true
);

U_opt_direct = [basis[i] ⋅ opt_states_direct[j] for i = 1:4, j = 1:4];
#-
gate_concurrence(U_opt_direct)
#jl @test round(gate_concurrence(U_opt_direct), digits=3) ≈ 1.0
#-
1 - unitarity(U_opt_direct)
#jl @test round(1 - unitarity(U_opt_direct), digits=3) ≈ 0.001
