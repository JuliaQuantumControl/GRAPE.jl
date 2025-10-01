```@meta
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: CC-BY-4.0

CollapsedDocStrings = false
```

# API

The stable public API of the `GRAPE` consists of following members:

* [`GRAPE.optimize`](@ref) as the main function to run an optimization
* [`GRAPE.GrapeResult`](@ref) as the object returned by [`GRAPE.optimize`](@ref), and accessible in callbacks
* [`QuantumControl.optimize`](@ref) with `method=GRAPE`, has a higher-level wrapper around [`GRAPE.optimize`](@ref) with extra features
* [`GRAPE.Trajectory`](@ref QuantumControl.Trajectory) as an alias of  [`QuantumControl.Trajectory`](@extref)
* [`GRAPE.set_default_ad_framework`](@ref QuantumControl.set_default_ad_framework) as an alias of [`QuantumControl.set_default_ad_framework`](@extref)

The remaining functions in `GRAPE` documented below should not be considered part of the stable API. They are guaranteed to be stable in bugfix (`x.y.z`) releases, but may change in feature releases (`x.y`).

Note that the `GRAPE` package does not _export_ any symbols. All members of the public API must be explicitly imported or used with their fully qualified name.

## [Index](@id api-index)

```@index
```

## [Reference](@id api-reference)

```@autodocs
Modules = [GRAPE]
```

```@docs
QuantumControl.Trajectory
QuantumControl.set_default_ad_framework
```
