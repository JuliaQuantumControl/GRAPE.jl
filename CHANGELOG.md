<!--
SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: CC-BY-4.0
-->

# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For releases pre-1.0, see the [GitHub Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases).


## [v1.2.0] — 2026-09-16

* Changed: The minimum supported Julia version is now 1.10 (LTS)
* Changed: The minimum supported versions of dependencies are now QuantumControl 0.11.5 and QuantumGradientGenerators 0.1.9. GRAPE does not work with QuantumGradientGenerators 0.1.8, which lacks the type-based `supports_inplace` trait of QuantumPropagators 0.9
* Changed: An `optimizer` from Optim.jl now requires Optim 2. Support for Optim 1 is dropped. Only first-order optimizers (e.g., `Optim.LBFGS()`) are accepted [[#114], replacing [#110]]
* Changed: The convergence tolerances of an Optim.jl `optimizer` use the keyword arguments `x_abstol`, `x_reltol`, `f_abstol`, `f_reltol`, and `g_abstol`, like `Optim.Options`. The previous `x_tol`, `f_tol`, and `g_tol` remain as aliases for `x_abstol`, `f_reltol`, and `g_abstol`
* Fixed: With an Optim.jl `optimizer`, the `callback` for the guess received iteration number 1 instead of 0, which shifted all iteration numbers by one and ran one iteration fewer than `iter_stop`
* Fixed: With an Optim.jl `optimizer`, `search_direction` and `step_width` for `Optim.ConjugateGradient` returned the search direction of the following iteration
* Fixed: A `callback` that mutates `pulsevals` for an Optim.jl `optimizer` now throws an error instead of silently desynchronizing the workspace from the optimizer


## [v1.1.0] — 2026-06-20

* Added: Support for state-dependent running costs [[#53], [#103], followup in [#105]]
* Fixed: Bug in `gradient_method = :taylor` incorrectly accessing `pulsevals`, leading to incorrect gradients when the control problem contains more than one control [fixed as part of [#103]]


## [v1.0.0] — 2025-10-30

Initial stable release. No breaking changes compared to [v0.8.1].

[Unreleased]: https://github.com/JuliaQuantumControl/GRAPE.jl/compare/v1.2.0..HEAD
[v1.2.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.2.0
[v1.1.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.1.0
[v1.0.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.0.0
[v0.8.1]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v0.8.1
[#114]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/114
[#110]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/110
[#105]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/105
[#103]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/103
[#53]: https://github.com/JuliaQuantumControl/GRAPE.jl/issues/53
