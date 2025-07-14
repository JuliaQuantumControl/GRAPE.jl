```@meta
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: CC-BY-4.0
```

# Usage

The `GRAPE` package is used in the context of the `QuantumControl` framework. You should be familiar with the [concepts used in the framework](@extref QuantumControl :label:`Glossary`) and its [overview](@extref QuantumControl :label:`Overview`).

For specific examples of the use of `GRAPE`, see the [Tutorials of the JuliaQuantumControl organization](https://juliaquantumcontrol.github.io/Tutorials/), e.g., the simple [State-to-state transfer in a two-level system](https://juliaquantumcontrol.github.io/Tutorials/TLS_State_to_State.html).

More generally:

* Set up a [`QuantumControl.ControlProblem`](@extref) with one or more [trajectories](@extref `QuantumControl.Trajectory`). The `problem` must have a set of controls, see [`QuantumControl.Controls.get_controls(problem)`](@extref QuantumControl `QuantumPropagators.Controls.get_controls`), that can be discretized as piecewise-constant on the intervals of the time grid, cf. [`QuantumPropagators.Controls.discretize_on_midpoints`](@extref).
* Make sure the `problem` includes a well-defined final time functional `J_T`. The GRAPE method also requires `chi` to determine the boundary condition ``\ket{\chi_k} = \partial J_T / \partial \bra{\Psi_k(T)}``. This can be determined automatically, analytically for known functions `J_T`, or via automatic differentiation, so it is an optional parameter.
* Propagate the system described by `problem` to ensure you understand the dynamics under the guess controls!
* Call [`QuantumControl.optimize`](@extref), or, preferably, [`QuantumControl.@optimize_or_load`](@extref `QuantumControl.Workflows.@optimize_or_load`) with `method = GRAPE`. Pass additional keyword arguments to customize GRAPE's behavior:

```@docs; canonical=false
QuantumControl.optimize(::ControlProblem, ::Val{:GRAPE})
```
