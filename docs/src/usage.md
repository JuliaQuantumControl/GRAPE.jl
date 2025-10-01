```@meta
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: CC-BY-4.0
```

# Usage

The `GRAPE` package is best used via the interface provided by the `QuantumControl` framework, see the [Relation to the QuantumControl Framework](@ref). It helps to be familiar with the [concepts used in the framework](@extref QuantumControl :label:`Glossary`) and its [overview](@extref QuantumControl :label:`Overview`).

The package can also be used standalone, as illustrated in the previous [Tutorial](@ref), and encapsulated in the API of the `GRAPE.optimize` function:

```@docs; canonical=false
GRAPE.optimize
```

## Relation to the QuantumControl Framework

The `GRAPE` package is associated with the broader [`QuantumControl` framework](@extref QuantumControl :doc:`index`). The role of `QuantumControl` in relation to `GRAPE` has two aspects:

1. `QuantumControl` provides a collection of components that are useful for formulating control problems in general, for solution via `GRAPE` or arbitrary other methods of quantum control. This includes, for example, [control functions](@extref QuantumControl `QuantumControlControlsAPI`) and [control amplitudes](@extref QuantumControl `QuantumControlAmplitudesAPI`), [data structures for time-dependent Hamiltonians or Liouvillians](@extref QuantumControl `QuantumControlGeneratorsAPI`), or [common optimization functionals](@extref QuantumControl `QuantumControlFunctionalsAPI`).

2. `QuantumControl` provides a common way to formulate a [`ControlProblem`](@extref `QuantumControl.ControlProblem`) and general [`optimize`](@extref `QuantumControl.optimize`) and [`@optimize_or_load`](@extref `QuantumControl.Workflows.@optimize_or_load`) functions that particular optimization packages like `GRAPE` can plug in to. The aim is to encourage a common interface between different optimization packages that makes it easy to switch between different methods.

```@docs; canonical=false
QuantumControl.optimize(::ControlProblem, ::Val{:GRAPE})
```
