```@meta
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: CC-BY-4.0

CurrentModule = GRAPE
```

# GRAPE.jl

```@eval
using Markdown
using Pkg

VERSION = Pkg.dependencies()[Base.UUID("6b52fcaf-80fe-489a-93e9-9f92080510be")].version

github_badge = "[![Github](https://img.shields.io/badge/JuliaQuantumControl-GRAPE.jl-blue.svg?logo=github)](https://github.com/JuliaQuantumControl/GRAPE.jl)"

version_badge = "![v$VERSION](https://img.shields.io/badge/version-v$VERSION-green.svg)"

Markdown.parse("$github_badge $version_badge")
```

Gradient Ascent Pulse Engineering in Julia

## Summary

The `GRAPE.jl` package implements Gradient Ascent Pulse Engineering [KhanejaJMR2005, FouquieresJMR2011, GoerzQ2022](@cite), a widely used method of [quantum optimal control](@extref QuantumControl :doc:`index`). The quantum state of a system can be described numerically by a complex vector ``\ket{\Psi(t)}`` that evolves under a differential equation of the form

```math
\def\ii{\mathrm{i}}
\begin{equation}\label{eq:tdse}
\ii \hbar \frac{\partial \ket{\Psi(t)}}{\partial t} = \hat{H}(\epsilon(t)) \ket{\Psi(t)}\,,
\end{equation}
```

where ``\hat{H}`` is a matrix (the [Hamiltonian](https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics)), most commonly) whose elements depend in some way on the control function ``\epsilon(t)``. We generally know the initial state of the system ``\ket{\Psi(t=0)}`` and want to find an ``\epsilon(t)`` that minimizes some functional ``J`` that depends on the states at some final time ``T``, as well as running costs on ``\ket{\Psi(t)}`` and values of ``\epsilon(t)`` at intermediate times.

The defining feature of the GRAPE method is that it considers ``\epsilon(t)`` as piecewise constant, i.e., as a vector of pulse values ``\epsilon_n``, for the ``n``'th interval of the time grid. This allows solving Eq. \eqref{eq:tdse} analytically for each time interval, and deriving an expression for the gradient ``\partial J / \partial \epsilon_n`` of the optimization functional with respect to the values of the control field. The pulse values are then updated based on the gradient, in an efficient scheme detailed in [Background](@ref GRAPE-Background) (or the ["TMIDR" short summary](@ref tmidr)).


## Contents

* [Home]()
    * [Related Software](@ref)
    * [Features](@ref)
    * [Contributing](@ref)
    * [History](@ref)

```@contents
Depth = 2
Pages = [pair[2] for pair in Main.PAGES[2:end-1]]
```


## Related Software

`GRAPE.jl` integrates with the [JuliaQuantumControl ecosystem](https://github.com/JuliaQuantumControl) and, in particular, the following packages:

* [`QuantumControl.jl`](https://github.com/JuliaQuantumControl/QuantumControl.jl) – The overall control framework, used to define the quantum control problem. `GRAPE` is used by calling [`QuantumControl.optimize`](@extref) with `method = GRAPE`.
* [`QuantumPropagators.jl`](https://github.com/JuliaQuantumControl/QuantumPropagators.jl) – The numerical backend for simulating the piecewise-constant time dynamics of the system. Implements efficient schemes such as Chebychev propagation [Tal-EzerJCP1984](@cite), but also can further delegate to [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/)
* [`QuantumGradientGenerators.jl`](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl) – Implementation of the gradient of a single-time-step evolution operator according to [GoodwinJCP2015](@citet), a [key component of the GRAPE scheme as implemented here](@ref Overview-Gradgen).

The GRAPE method ("discretize first") compares most directly to Krotov's method ("derive optimality first, discretize second") [GoerzSPP2019](@cite), implemented in [`Krotov.jl`](https://github.com/JuliaQuantumControl/Krotov.jl) with a compatible interface.


### Other implementations of GRAPE

There have been a number of implementations of the GRAPE method in different contexts.

In the context of [NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance):

* [`SIMPSON`](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson) – Simulation program for solid-state NMR spectroscopy (C) [TosnerJMR2009](@cite)
* [`Spinach`](https://spindynamics.org/?page_id=12) – Spin dynamics simulation library (Matlab) [HogbenJMR2011](@cite)
* [`pulse-finder`](https://github.com/caryan/pulse-finder/) – Matlab code for GRAPE optimal control in NMR [pulse-finder](@cite)

More recent implementations, geared towards more general purposes like quantum information:

* [`QuTIP`](https://qutip.org) – Quantum Toolbox in Python [JohanssonCPC2013](@cite)
* [`C3`](https://github.com/q-optimize/c3) – Toolset for control, calibration, and characterization of physical systems (Python) [WittlerPRA2021](@cite)
* [`QuOCS`](https://github.com/Quantum-OCS/QuOCS) – Python software package for model- and experiment-based optimizations of quantum processes [RossignoloCPC2023](@cite)
* [`QuanEstimation`](https://github.com/QuanEstimation/QuanEstimation) – Python-Julia-based open-source toolkit for quantum parameter estimation [ZhangPRR2022](@cite)

As a direct precursor to `GRAPE.jl`:

* [`QDYN`](https://www.qdyn-library.net) – Fortran 95 library and collection of utilities for the simulation of quantum dynamics and optimal control with a focus on both efficiency and precision


## Features

The `GRAPE.jl` package aims to avoid common shortcomings in [existing implementations](@ref "Other implementations of GRAPE") by emphasizing the following design goals and features:

* **Performance** similar to that of Fortran [QDYN](@cite), allowing to extend to quantum systems of large dimension. The numerical cost of the GRAPE method is dominated by the cost of evaluating the time evolution of the quantum system. `GRAPE.jl` delegates this to efficient piecewise-constant propagators in [`QuantumPropagators.jl`](https://github.com/JuliaQuantumControl/QuantumPropagators.jl) or the general [`DifferentialEquations.jl` framework](https://docs.sciml.ai/DiffEqDocs/stable/) [RackauckasJORS2017](@cite).

* **Generality** through the adoption of the [concepts defined in `QuantumControl.jl`](@extref QuantumControl `Glossary`)

  * Allow functionals that depend on an arbitrary set of "trajectories" ``\{ |\Psi_k(t)⟩\}``, each evolving under a potentially different ``\hat{H}_k``. In contrast to the common restriction to a single state ``|\Psi⟩`` or a single unitary ``\hat{U}`` as the dynamical state, this enables ensemble optimization for robustness against noise, e.g., Ref. [GoerzPRA2014](@cite). The optimization over multiple trajectories is parallelized.
  * Each ``\hat{H}_k`` may depend on an arbitrary number of controls ``\epsilon_l(t)`` in an arbitrary way, with a distinction between [time-dependent "amplitudes"](@extref QuantumControl `Control-Amplitude`) and ["controls"](@extref QuantumControl `Control-Function`), going beyond the common assumption of linear controls, ``\hat{H} = \hat{H}_0 + \epsilon(t) \hat{H}_1``.

* **Flexibility** to work, via [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY), with any custom data structures for quantum states ``|\Psi_k(t)⟩`` or the dynamic generators ``\hat{H}_k(\{\epsilon_l(t)\})``, enabling a wide range of applications, from NMR spin systems to superconducting circuits or trapped atoms in quantum computing, to systems with spatial degrees of freedom, e.g., Ref. [DashAVSQS2024](@cite). This also includes open quantum systems.

* **Arbitrary Functionals** via semi-automatic differentiation [GoerzQ2022](@cite). The numerical scheme implemented in `GRAPE.jl` is derived from a generalization that calculates the gradient of the final-time functional via the chain rule with respect to the states ``|\Psi_k(T)⟩``. This allows going beyond "standard functionals" based on overlaps with target states [KhanejaJMR2005, PalaoPRA2003](@cite) to any computable functional, including non-analytic functionals such as entanglement measures [KrausPRA2001, WattsPRA2015, GoerzPRA2015](@cite). The consequence of this generalization is a boundary condition for the backward-propagation as ``|\chi_k(T)⟩ = -\partial J_T/\partial ⟨\Psi_k(T)|`` instead of the target state in the traditional scheme. The state ``|\chi_k(T)⟩``, as well as derivatives for any running costs, can optionally be obtained via [automatic differentiation](https://juliadiff.org).


`GRAPE.jl` is aimed at researchers in quantum control wanting flexibility to explore novel applications, while also requiring high numerical performance.


## Contributing

See [`CONTRIBUTING.md`](https://github.com/JuliaQuantumControl/.github/blob/master/CONTRIBUTING.md#contributing-to-juliaquantumcontrol-packages).

Consider using the [JuliaQuantumControl Dev Environment](https://github.com/JuliaQuantumControl/JuliaQuantumControl?tab=readme-ov-file#juliaquantumcontrol-dev-environment).

## History

See the [Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases) on Github.
