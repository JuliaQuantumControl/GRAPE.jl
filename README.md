<!--
SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: CC-BY-4.0
-->

# GRAPE.jl

[![Version](https://juliahub.com/docs/General/GRAPE/stable/version.svg)](https://juliahub.com/ui/Packages/General/GRAPE)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaquantumcontrol.github.io/GRAPE.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaquantumcontrol.github.io/GRAPE.jl/dev)
[![JOSS](https://joss.theoj.org/papers/25e7a240c129459ad160dd3fb9d009d8/status.svg)](https://joss.theoj.org/papers/25e7a240c129459ad160dd3fb9d009d8)
[![Build Status](https://github.com/JuliaQuantumControl/GRAPE.jl/workflows/CI/badge.svg)](https://github.com/JuliaQuantumControl/GRAPE.jl/actions)
[![REUSE](https://github.com/JuliaQuantumControl/GRAPE.jl/actions/workflows/REUSE.yml/badge.svg)](https://github.com/JuliaQuantumControl/GRAPE.jl/actions/workflows/REUSE.yml)
[![Coverage](https://codecov.io/gh/JuliaQuantumControl/GRAPE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaQuantumControl/GRAPE.jl)

Implementation of GRadient Ascent Pulse Engineering (GRAPE)

Part of the [JuliaQuantumControl] organization, to be used in conjunction with the [`QuantumControl.jl`] framework.


## Installation

[As usual for a registered Julia package](https://docs.julialang.org/en/v1/stdlib/Pkg/), `GRAPE` can be installed by typing

```
] add GRAPE
```

in the Julia REPL.

## Usage

```julia
using QuantumControl
using GRAPE

# Set up a `QuantumControl.ControlProblem`

result = QuantumControl.optimize(problem; method=GRAPE)
```

See the [Usage section in the Documentation](https://juliaquantumcontrol.github.io/GRAPE.jl/stable/usage/)

## Documentation

The documentation of `GRAPE.jl` is available at <https://juliaquantumcontrol.github.io/GRAPE.jl>.

For a broader perspective, also see the [documentation of the `QuantumControl.jl` framework](https://juliaquantumcontrol.github.io/QuantumControl.jl/).

## Contributing

See [`CONTRIBUTING.md`](https://github.com/JuliaQuantumControl/.github/blob/master/CONTRIBUTING.md#contributing-to-juliaquantumcontrol-packages) and the [organization development notes](https://github.com/JuliaQuantumControl#development).


[`QuantumControl.jl`]: https://github.com/JuliaQuantumControl/QuantumControl.jl#readme
[JuliaQuantumControl]: https://github.com/JuliaQuantumControl

## History

See the [`CHANGELOG.md`](CHANGELOG.md) and the [Release Notes](https://github.com/JuliaQuantumControl/GRAPE.jl/releases).

## License

The source code of this project is licensed under the [MIT License](LICENSE). The documentation is licensed under [Creative Commons (`CC-BY-4.0`)](https://creativecommons.org/licenses/by/4.0/deed.en). License information for all files is [automatically tracked](https://github.com/JuliaQuantumControl/GRAPE.jl/actions/workflows/REUSE.yml) according to [REUSE](https://reuse.software) and can be verified using the [`reuse` tool](https://github.com/fsfe/reuse-tool?tab=readme-ov-file#reuse), e.g., by running `reuse spdx`.
