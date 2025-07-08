# GRAPE.jl

[![Version](https://juliahub.com/docs/General/GRAPE/stable/version.svg)](https://juliahub.com/ui/Packages/General/GRAPE)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaquantumcontrol.github.io/GRAPE.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaquantumcontrol.github.io/GRAPE.jl/dev)
[![Build Status](https://github.com/JuliaQuantumControl/GRAPE.jl/workflows/CI/badge.svg)](https://github.com/JuliaQuantumControl/GRAPE.jl/actions)
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

See the [Usage section in the Documentation](https://juliaquantumcontrol.github.io/GRAPE.jl/dev/usage/)

## Documentation

The documentation of `GRAPE.jl` is available at <https://juliaquantumcontrol.github.io/GRAPE.jl>.

For a broader perspective, also see the [documentation of the `QuantumControl.jl` framework](https://juliaquantumcontrol.github.io/QuantumControl.jl/).

## Contributing

See [`CONTRIBUTING.md`](https://github.com/JuliaQuantumControl/.github/blob/master/CONTRIBUTING.md#contributing-to-juliaquantumcontrol-packages) and the [organization development notes](https://github.com/JuliaQuantumControl#development).


[`QuantumControl.jl`]: https://github.com/JuliaQuantumControl/QuantumControl.jl#readme
[JuliaQuantumControl]: https://github.com/JuliaQuantumControl
