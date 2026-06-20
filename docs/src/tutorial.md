```@meta
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: CC-BY-4.0
ShareDefaultModule = true
```

# Tutorial

This introductory tutorial illustrates the basic usage of the `GRAPE` package for an example of a quantum gate on two qubits. It tries to be a self-contained as possible, to highlight some of the core concepts and features of the `GRAPE` package:

* Using arbitrary data structures for quantum states and operators
* Defining a control problem based on multiple "trajectories", with multi-threading
* Optimizing multiple control fields at the same time (the real and imaginary part of a physical control)
* Using non-trivial ["control amplitudes"](@extref QuantumControl `Control-Amplitude`) to ensure smooth switch-on/off
* Using automatic differentiation to minimize arbitrary optimization functionals
* Applying bounds on the amplitude of the control field


## A Two-Qubit System

The quantum state of a [two-qubit system](https://en.wikipedia.org/wiki/Qubit) is described by a complex vector spanned by the possible classical bit configurations, ``|00⟩``, ``|01⟩``, ``|10⟩``, and ``|11⟩`` in [braket notation](https://en.wikipedia.org/wiki/Bra–ket_notation). This basis is represented as


```@example
using StaticArrays: SVector

basis = [
    SVector{4}(ComplexF64[1, 0, 0, 0]),  # |00⟩
    SVector{4}(ComplexF64[0, 1, 0, 0]),  # |01⟩
    SVector{4}(ComplexF64[0, 0, 1, 0]),  # |10⟩
    SVector{4}(ComplexF64[0, 0, 0, 1]),  # |11⟩
];
nothing # hide
```

An arbitrary vector ``|Ψ⟩ = α_{00} |00⟩ + α_{01} |01⟩ + α_{10} |10⟩ + α_{11} |11⟩`` with complex coefficients ``α_i = α_{00}``, ``α_{01}``, ``α_{10}``, ``α_{11}`` is understood according to the [Born rule](https://en.wikipedia.org/wiki/Born_rule) to specify the probability as ``|α_{i}|^2`` to find the system in the corresponding possible classical state on measurement.

We've used the [`StaticArrays` Julia package](https://github.com/JuliaArrays/StaticArrays.jl) to encode the quantum states as an [`SVector`](@extref StaticArrays), which provides numerical advantages for very small vectors such as these, but also illustrates an important design choice of the `GRAPE` package: states can be expressed in any data structure fulfilling a [well-specified interface](@extref QuantumControl `QuantumPropagators.Interfaces.check_state`). This allows for custom, problem-specific encodings.

Prior to a measurement, quantum mechanics postulates that a quantum state ``|Ψ(t)⟩`` evolves according to the [Schrödinger equation](https://en.wikipedia.org/wiki/Schrödinger_equation#Time-dependent_equation), based on a Hamiltonian matrix ``\hat{H}``. We consider here a setup inspired by [superconducting transmon qubits](https://en.wikipedia.org/wiki/Superconducting_quantum_computing#Transmon). Each qubit is driven by an oscillating microwave field with frequency near the resonance for that qubit, and the two qubits are coupled by a shared transmission line. As is common in simulating quantum systems with a fast-oscillating field, the [rotating wave approximation](https://en.wikipedia.org/wiki/Rotating-wave_approximation) allows us to formulate the system in the "rotating frame" of the microwave field center frequency ``ω_{mw}``. The Hamiltonian for the two-qubit system is then

```math
\hat{H}
    = \left(δ_1 \hat{n}_1 + \frac{Ω(t)}{2} \hat{a}_1 + \frac{Ω^*(t)}{2} \hat{a}_1^\dagger\right)
    +  \left(δ_2 \hat{n}_2 + \frac{Ω(t)}{2} \hat{a}_2 + \frac{Ω^*(t)}{2} \hat{a}_2^\dagger\right)
    + J (\hat{a}_1\hat{a}_2^\dagger + \hat{a}_1^\dagger\hat{a}_2)
    \tag{1}
```

with the usual [raising and lowering operators](https://en.wikipedia.org/wiki/Creation_and_annihilation_operators#Matrix_representation) ``\hat{a}^\dagger`` and ``\hat{a}`` and the number operator ``\hat{n} = \hat{a}^\dagger \hat{a}``, and where $δ_{1,2}$ is the [detuning](https://en.wikipedia.org/wiki/Laser_detuning) of microwave angular central frequency from the transition frequency of the first and second qubit, respectively, ``J`` is the static coupling, and ``Ω(t)`` is the envelope of the microwave field. In the rotating frame, when the physical microwave can deviate from its central frequency ``ω_{mw}``, this is equivalent to a _complex_ amplitude, where the complex phase of ``Ω(t)`` is the phase relative to ``ω_{mw} t``. Note that Eq. (1) is an oversimplified model for _actual_ transmons, which should include more levels than just the two lowest-lying levels defining the qubit.

## Units

Generally, the elements of the Hamiltonian are in units of energy. However, the [`QuantumPropagators` package](@extref QuantumPropagators :doc:`index`) that `GRAPE` uses to simulate all dynamics assumes ``ħ = 1``, turning all elements of the Hamiltonian into angular frequencies (from the [Planck relation](https://en.wikipedia.org/wiki/Planck_relation) ``E = ħω``), expressed in units of 2π⋅Hz. It also makes "energy" and "time" directly reciprocal (only the phase ``ωt`` is relevant for the dynamics). In numerics generally, it is best for all quantities to have magnitudes between maybe 10⁻³ and 10³ to avoid floating point errors. Here, the numerical quantities are the elements of the operators and the values in the time grid. With superconducting qubits typically having energies in a GHz regime (with detunings in MHz) and a timescale of ns for operations, we can define corresponding "internal" units:

```@example
const GHz = 1.0;
const MHz = 0.001GHz;
const ns = 1.0;
nothing # hide
```

We can then specify any of the "energy" parameters below with units, e.g.,

```@example
using LinearAlgebra: ⋅  # just so that the code looks nicer
δ₁ = 100⋅2π⋅MHz;
nothing # hide
```

## Control Amplitudes

``Ω(t)`` in Eq. (1) is our control amplitude; the aim of optimal control is to find the particular ``Ω(t)`` that steers the system in some particular way. In our case: to implement a quantum gate. However, the implementation of optimal control in `GRAPE` assumes real-valued controls. Thus, we have to split the complex ``Ω(t)`` into an independent real and imaginary part.

We may also want to place some _physical_ constraints onto ``Ω(t)``. For example, we may want ``Ω(t)`` to smoothly switch on from zero and off to zero at the beginning and end of the time grid. One way of achieving this is to define ``Ω(t) = S(t) ϵ(t)`` where ``S(t)`` is a static shape that has the smooth switch-on and off, and ``ϵ(t)`` is an arbitrary function that we can optimize freely.

The `QuantumPropagators` framework provides a [`ShapedAmplitude`](@extref `QuantumPropagators.Amplitudes.ShapedAmplitude`) object to implement this:

```@example
using QuantumPropagators.Amplitudes: ShapedAmplitude
```

For a fixed time grid ending at

```@example
T = 400ns;
nothing # hide
```

we can define the function ``S(t)`` as

```@example
using QuantumPropagators.Shapes: flattop
shape(t) = flattop(t, T = T, t_rise = 15ns);
nothing # hide
```

where [`flattop`](@extref `QuantumPropagators.Shapes.flattop`) is a function that switches on smoothly from zero, reaches 1.0 after 15 ns, and remains constant at 1.0 until the last 15 ns before the final time ``T =`` 400 ns, where it smoothly switches off to zero.

We can then define an initial guess for ``ϵ(t)`` as a constant function and assemble that into the complete ``Ω(t)``:

```@example
ϵ_re_guess(t; Ω₀ = 35⋅2π⋅MHz) = Ω₀
ϵ_im_guess(t) = 0.0

Ω_re_guess = ShapedAmplitude(ϵ_re_guess; shape);
Ω_im_guess = ShapedAmplitude(ϵ_im_guess; shape);
nothing # hide
```

By setting the guess for the imaginary part to zero, we let the system start exactly at the central frequency of the microwave field.

In general, the choice of the initial guess function can have a significant impact on the optimization. What makes a good guess pulse, in terms of which shape to use, or what initial amplitude (35⋅2π MHz, here) is often the result of some physical intuition, or some trial and error.

With the above definition, `Ω_re_guess` and `Ω_im_guess` are ["control amplitudes"](@extref QuantumControl `Control-Amplitude`) that are the _physical_ control, whereas `ϵ_re_guess` and `ϵ_im_guess` are the ["controls"](@extref QuantumControl `Control-Function`) from the perspective of GRAPE. The basic task of the GRAPE method is to discretize these controls to the intervals of the time grid (["pulses"](@extref QuantumControl `Pulse`), and then iteratively update the pulse values at each time interval, based on the gradient of some optimization functional with respect to the pulse value. This distinction between physical control amplitudes and optimization control functions / pulses is a core design aspect of `GRAPE`, providing great flexibility in the models that can be used for optimization.


We can now define an explicit time grid

```@example
tlist = collect(range(0, T, step = 0.1ns));
nothing # hide
```

and plot the combined ``Ω(t)``:

````@example
using Plots
ENV["GKSwstype"] = "100" # hide
gr() # hide
using QuantumPropagators.Controls: discretize

"""
Plot the given complex pulse in two panels: absolute value and phase.

```julia
fig = plot_complex_pulse(tlist, Ω; time_unit=:ns, ampl_unit=:(2π⋅MHz), kwargs...)
```

generates a plot of the complex field `Ω` over the time grid `tlist`.

Arguments:

* `tlist`: A vector of time grid values
* `Ω`: A complex vector of the same length as `tlist` or a function `Ω(t)` returning a complex number
* `time_unit`: A symbol that evaluates to the conversion factor for the time unit
* `ampl_unit`: A symbol that evaluates to the conversion factor of the amplitude unit

All other keyword arguments are forwarded to `Plots.plot`.
```
"""
function plot_complex_pulse(tlist, Ω; time_unit=:ns, ampl_unit=:(2π⋅MHz), kwargs...)

    # make sure Ω is defined on *points* of `tlist`; since `discretize`
    # returns real values, we handle the real and imaginary parts separately
    if Ω isa Function
        Ω = discretize(t -> real(Ω(t)), tlist) .+ 1im .* discretize(t -> imag(Ω(t)), tlist)
    else
        Ω = discretize(real.(Ω), tlist) .+ 1im .* discretize(imag.(Ω), tlist)
    end

    s_ampl_unit = string(ampl_unit)
    if startswith(s_ampl_unit, "(2π) ⋅ ")
        s_ampl_unit = "2π " * s_ampl_unit[11:end]
    end

    ax1 = plot(
        tlist ./ eval(time_unit),
        abs.(Ω) ./ eval(ampl_unit);
        label="|Ω|",
        xlabel="time ($time_unit)",
        ylabel="amplitude ($s_ampl_unit)",
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

const 𝕚 = 1im

fig = plot_complex_pulse(tlist, t -> Ω_re_guess(t) + 𝕚 * Ω_im_guess(t))
using DisplayAs; fig |> DisplayAs.SVG # hide
````

We see the expected shape with the smooth switch-on/-off and the amplitude of 35⋅2π MHz that we specified, while from the perspective of the GRAPE method, the "guess control" is a constant function.

## Hamiltonian

With the separation of ``Ω(t)`` into real and imaginary parts, we can now express the Hamiltonian in Eq. (1) numerically:


````@example
import QuantumPropagators: hamiltonian
using StaticArrays: SMatrix

"""Construct the time-dependent Hamiltonian for a two-qubit transmon system.

```julia
Ĥ = transmon_hamiltonian(; δ₁, δ₂, J, Ω_re, Ω_im)
```

constructs a [Generator](@extref `QuantumPropagators.Generators.Generator`) for
a two-transmon system truncated to two levels for each qubit, in the rotating
wave approximation (RWA)

Arguments:

* `δ₁`: The detuning of qubit 1 from the RWA angular frequency
* `δ₂`: The detuning of qubit 2 from the RWA angular frequency
* `J`: The static coupling between the two qubits, as angular frequency
* `Ω_re`: The amplitude (angular frequency) of the real part of the microwave drive
* `Ω_im`: The amplitude (angular frequency) of the imaginary part of the microwave drive

Both `Ω_re` and `Ω_im` can be given as a function `Ω_re(t)` and `Ω_im(t)`, or as a
vector of values on the time grid, or on the intervals of the time grid.
"""
function transmon_hamiltonian(; δ₁, δ₂, J, Ω_re, Ω_im)

    Ĥ₀ = SMatrix{4,4,ComplexF64}(
        [0        0       0       0
         0        δ₂      J       0
         0        J       δ₁      0
         0        0       0       δ₁+δ₂]
    )

    Ĥ₁_re = (1/2) * SMatrix{4,4,ComplexF64}(
        [0      1       1      0
         1      0       0      1
         1      0       0      1
         0      1       1      0]
    )

    Ĥ₁_im = (𝕚/2) * SMatrix{4,4,ComplexF64}(
        [0      -1      -1      0
         1       0       0     -1
         1       0       0     -1
         0       1       1      0]
    )

    return hamiltonian(Ĥ₀, (Ĥ₁_re, Ω_re), (Ĥ₁_im, Ω_im))

end
nothing # hide
````

where we have used the [`QuantumPropagators.Generators.hamiltonian`](@extref) function to construct an object that serves as a time-dependent Hamiltonian (a ["Generator"](@extref QuantumControl :label:`Generator`), in the terminology of the general `QuantumControl` framework, ensuring adherence to the [required interface](@extref QuantumControl `QuantumControl.Interfaces.check_generator`)). The [`SMatrix`](@extref StaticArrays) used for the drift and control Hamiltonian operators match the [`SVector`](@extref StaticArrays) used to encode states.

We use some arbitrary but reasonable values here, with the central frequency of the microwave field centered between the frequencies of the two qubits, resulting in a detuning of ±100⋅2π MHz and a static coupling of 3⋅2π MHz.


```@example
Ĥ = transmon_hamiltonian(;
    δ₁ = 100⋅2π⋅MHz,
    δ₂ = -100⋅2π⋅MHz,
    J = 3⋅2π⋅MHz,
    Ω_re = Ω_re_guess,
    Ω_im = Ω_im_guess,
)
```


## Optimization Target

We will now use the GRAPE method to find an ``Ω(t)`` that implements a [CNOT](https://en.wikipedia.org/wiki/Controlled_NOT_gate), one of the fundamental building blocks of a quantum computer. In classical terms, a CNOT gate flips the state of the second qubit if and only if the first qubit is "1". This is fundamentally an [entangling](https://en.wikipedia.org/wiki/Quantum_entanglement) operation, and the equivalent of an "if" statement in a quantum circuit.

In terms of the `basis` states, we have corresponding target states:


```@example
target_states = [
    SVector{4}(ComplexF64[1, 0, 0, 0]),  # |00⟩
    SVector{4}(ComplexF64[0, 1, 0, 0]),  # |01⟩
    SVector{4}(ComplexF64[0, 0, 0, 1]),  # |10⟩ → |11⟩
    SVector{4}(ComplexF64[0, 0, 1, 0]),  # |11⟩ → |10⟩
];
nothing # hide
```

Beyond a classical interpretation, _any_ quantum state ``|Ψ⟩`` should be mapped to the state

```math
|Ψ^{(tgt)}⟩ = \hat{O} |Ψ⟩ = \sum_k α_k \hat{O} |k⟩
```

for the four basis states ``|k⟩ = |00⟩``, ``|01⟩``, ``|10⟩``, ``|11⟩``, with

```math
\hat{O} ≡ \begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{pmatrix}\,.
```

being the matrix representation of the CNOT gate.


We can use the [`QuantumPropagators` propagators](@extref QuantumPropagators :doc:`index`) to simulate ("propagate") the ``|10⟩`` state exemplarily. `QuantumPropagators` can use a variety of methods to do this. We use the [`Cheby` method](@extref QuantumPropagators `method_cheby`) here, which is extremely efficient at solving the Schrödinger equation for piecewise-constant pulses by expanding the [time evolution operator](https://en.wikipedia.org/wiki/Time_evolution#Time-independent_Hamiltonian) into Chebychev polynomials.

```@example
using QuantumPropagators: propagate, Cheby

pops = propagate(
    basis[3], Ĥ, tlist; method=Cheby, storage=true, observables=(Ψ -> Array(abs2.(Ψ)), )
)

fig = plot(tlist ./ ns, pops'; label=["00" "01" "10" "11"], xlabel="time (ns)", legend=:outertop, legend_column=-1)
using DisplayAs; fig |> DisplayAs.SVG # hide
```

From the dynamics, we can see that the ``|10⟩`` initial state remains approximately unchanged by the guess controls, instead of evolving into the ``|11⟩`` state as it should.

In order to find a field ``\Omega(t)`` that implements the desired CNOT gate, the GRAPE method minimizes an appropriate (user-supplied) mathematical functional. We define here


````@example
@doc raw"""Square-Modulus functional

```julia
J_T = J_T_sm(Ψ, trajectories)
```

calculates the real-valued scalar

```math
J_{T,sm} = 1 - \frac{1}{N^2} \left\vert\sum_n ⟨Ψ_k(T)|Ψ_k^{(tgt)}⟩\right\vert^2
```

where the state ``|Ψ_k⟩`` is the k'th element of `Ψ` and ``|Ψ_k^{(tgt))⟩`` is
the state stored in the `target_state` attribute of the k'th element of the
`trajectories` list. The ``|Ψ_k⟩`` should generally be the states obtained from
propagating the state stored in the `initial_state` attribute of the k'th
element of `trajectories` forward to some final time ``T``.

Conceptually, this becomes zero if and only if all the forward propagated
states have an overlap of 1 with their respective target state, up to a
global phase.
"""
function J_T_sm(Ψ, trajectories)
    N = length(trajectories)
    f = zero(ComplexF64)
    for (Ψ_k, traj) in zip(Ψ, trajectories)
        f += Ψ_k ⋅ traj.target_state
    end
    return 1.0 - (abs2(f) / N^2)
end

nothing # hide
````

to implement

```math
J_{T,sm} = 1 - \frac{1}{N^2} \left\vert\sum_n ⟨Ψ_k(T)|Ψ_k^{(tgt)}⟩\right\vert^2
```

appropriate for optimization of a quantum gate if the states ``|Ψ_k(T)⟩`` are evolved from the ``N = 4`` basis states (for a two-qubit gate) and the target states ``|Ψ_k^{(tgt)}⟩`` are chosen as ``\hat{O} |Ψ_k(T)⟩``. A generalization of the functional `J_T` (which is one of the standard functionals of quantum control) is also implemented in [`QuantumControl.Functionals.J_T_sm`](@extref). We have re-implemented it here by hand to illustrate the API that the `GRAPE` package expects for a functional `J_T`. In particular, `GRAPE` requires optimization functionals to be defined in terms of `trajectories`, as the way to implicitly define the final-time states ``|Ψ_k(T)⟩`` that enter the functional:


```@example
using GRAPE: Trajectory

trajectories = [
    Trajectory(Ψ, Ĥ; target_state=Ψ_tgt)
    for (Ψ, Ψ_tgt) in zip(basis, target_states)
]
```

Each trajectory, at a minimum, defines an initial state `Ψ` and a dynamical generator `Ĥ` (i.e., a time-dependent Hamiltonian, in our case). In general, these can be arbitrary problem-specific objects, as long as they implement the [required interfaces](@extref QuantumControl `QuantumPropagatorsInterfacesAPI`). Each trajectory can also attach arbitrary additional attributes that may be used by the `J_T` function, e.g., a `target_state` in our case.

Defining an optimization in terms of multiple "trajectories" has a number of benefits. For one, simulating the dynamics for the different trajectories can be performed in parallel, resulting in a potential speedup proportional to the number of trajectories. Second, it also enables advanced use cases such as "ensemble optimizations" that consider multiple "copies" of the quantum system, each with a different noisy Hamiltonian, in an effort to find controls that are robust with respect to these variations.

The important point is that all of the trajectories _share_ the same set of controls, two in our case (the real and imaginary part of ``Ω(t)`` divided by our shape ``S(t)``):


```@example
using QuantumPropagators.Controls: get_controls

get_controls(trajectories)
```


The GRAPE method works by numerically evaluating gradients of the given functional `J_T` with respect to the pulse values at each interval of the time grid. As derived fully in [Background](@ref GRAPE-Background), the resulting scheme depends explicitly on `J_T` via the definition of a set of "adjoint states",

```math
|χ_k(T)⟩ \equiv - \frac{\partial J_T}{\partial \bra{Ψ_k(T)}}\,,
```

The calculation of these states must be implemented in a function `chi` that will be passed to the optimization alongside `J_T`. For ``J_{T,sm}``, the states ``|χ_k(T)⟩`` can be calculated analytically, as

```math
|χ_k(T)⟩ = \frac{1}{N^2} \left(\sum_j ⟨Ψ_j^{(tgt)}|Ψ_j(T)⟩\right) |Ψ_k^{(tgt)}⟩\,,
```

and is implemented in [`QuantumControl.Functionals.chi_sm`](@extref).

However, for more advanced functionals, `J_T` may not always have an analytic derivative that can be written in closed form, or the derivative is just too cumbersome to write out. The GRAPE package allows constructing `chi` via automatic differentiation, e.g., via the popular [`Zygote` package](@extref Zygote :doc:`index`). This feature can be enabled as follows:


```@example
import Zygote
using GRAPE

GRAPE.set_default_ad_framework(Zygote)
```

## Optimization

We can now run the actual optimization by calling `GRAPE.optimize`. The main positional arguments are the `trajectories` that enter the functional and the time grid `tlist`. Beyond that, the _required_ key arguments are `J_T` to give the functional and `prop_method` to specify the method to be used for any internal time propagation. We do not specify `chi`, since we are relying on the automatic differentiation that we activated above.

The optimization will run for as many iterations as given by `iter_stop` (5000 by default), and we can give a `check_convergence` callback to stop the optimization earlier, based on some value of the functional being reached. There is also a `callback` to print some information after each iteration.

By setting `use_threads=true`, we parallelize the optimization over the different `trajectories`. This requires that the Julia process was started with the `-t <NTHREADS>` option, or that the `JULIA_NUM_THREADS` environment variable was set before starting the Julia process.

Lastly, we make use of the `GRAPE` package's ability to apply box constraints, i.e., an `upper_bound` and `lower_bound` for the values of the control pulse.

```@example

result = GRAPE.optimize(
    trajectories,
    tlist;
    prop_method = Cheby,
    J_T = J_T_sm,
    callback = GRAPE.make_grape_print_iters(),
    iter_stop = 200,
    check_convergence = (res -> ((res.J_T < 1e-2) && "Gate error < 10⁻²")),
    upper_bound = 50⋅2π⋅MHz,
    lower_bound = -50⋅2π⋅MHz,
    use_threads = true,
)
nothing # hide
```

```@example
result # hide
```

## Optimization Result

The call to `GRAPE.optimize` returns a results-object from which we can extract the optimized controls:

```@example
ϵ_opt = result.optimized_controls[1] + 𝕚 * result.optimized_controls[2];
nothing # hide
```

While the guess controls `ϵ_re_guess` and `ϵ_im_guess` that we defined when setting up the system Hamiltonian could have been either functions (as they were) or vectors of pulse values, the GRAPE method inherently discretizes all control to a time grid (the method is piecewise-constant, by definition). Thus, the `optimized_controls` are vectors of control values for each point in `tlist`

Also remember that we used a `ShapedAmplitude` when setting up the Hamiltonian, with the definition ``Ω(t) = S(t)ϵ(t)``. Thus, to get the _physical_ optimized control field, we need to multiply with that same shape `S(t)`, now also discretized to the time grid.

```@example
Ω_opt = ϵ_opt .* discretize(Ω_re_guess.shape, tlist)

fig = plot_complex_pulse(tlist, Ω_opt)
using DisplayAs; fig |> DisplayAs.SVG # hide
```

We can see how the optimization has tuned both the amplitude and the complex phase of the microwave field, while retaining the overall shape ``S(t)``. We can also simulate again the dynamics of the ``|10⟩`` state under the optimized fields:

```@example
Ĥ_opt = transmon_hamiltonian(;
    δ₁ = 100⋅2π⋅MHz,
    δ₂ = -100⋅2π⋅MHz,
    J = 3⋅2π⋅MHz,
    Ω_re = real.(Ω_opt),
    Ω_im = imag.(Ω_opt),
)

pops = propagate(
    basis[3], Ĥ_opt, tlist; method=Cheby, storage=true, observables=(Ψ -> Array(abs2.(Ψ)), )
)

fig = plot(tlist ./ ns, pops'; label=["00" "01" "10" "11"], xlabel="time (ns)", legend=:outertop, legend_column=-1)
using DisplayAs; fig |> DisplayAs.SVG # hide
```

As expected (and demanded by the "conditional NOT"), ``|10⟩`` now evolves into ``|11⟩``.
