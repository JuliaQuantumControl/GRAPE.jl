<!--
SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: CC-BY-4.0
-->

# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For releases pre-1.0, see the [GitHub Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases).

## [Unreleased]

* Added: Support for state-dependent running costs [[#53], [#103]]
* Fixed: Bug in `gradient_method = :taylor` incorrectly accessing `pulsevals`, leading to incorrect gradients when the control problem contains more than one control [fixed as part of [#103]]

## [v1.0.0] — 2025-10-30

Initial stable release. No breaking changes compared to [v0.8.1].

[Unreleased]: https://github.com/JuliaQuantumControl/GRAPE.jl/compare/v1.0.0..HEAD
[v1.0.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.0.0
[v0.8.1]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v0.8.1
[#103]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/103
[#53]: https://github.com/JuliaQuantumControl/GRAPE.jl/issues/53
