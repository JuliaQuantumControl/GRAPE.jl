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

## Installation

[As usual for a registered Julia package](https://docs.julialang.org/en/v1/stdlib/Pkg/), `GRAPE` can be installed by typing

```
] add GRAPE
```

in the Julia REPL.

## Usage Example

A minimal working example optimizing a state-to-state transition `|0⟩ → |1⟩` in a two-level quantum system:

```julia
using GRAPE

using QuantumPropagators: hamiltonian  # data structure for `H = H₀ + ϵ(t) H₁`
using QuantumControl.Functionals: J_T_sm  # square-modulus functional
using QuantumPropagators: ExpProp  # propagation method: matrix exponentiation

ϵ(t) = 0.2 # guess pulse

H = hamiltonian([1  0; 0 -1], ([0  1; 1  0], ϵ))  # time-dependent Hamiltonian
ket_0, ket_1 = ComplexF64[1, 0], ComplexF64[0, 1]  # basis states |0⟩, |1⟩
tlist = collect(range(0, 5, length=501));  # time grid; final time T = 5.0

# Optimization functionals depend on states |Ψ(T)⟩, described by a "trajectory"
traj = GRAPE.Trajectory(
    initial_state = ket_0,
    generator = H,
    target_state = ket_1
);

result = GRAPE.optimize(
    [traj], tlist;
    prop_method = ExpProp,  # suitable for small systems only!
    J_T = J_T_sm,  #  J_T = 1 - |⟨Ψ(T)|1⟩|²
    check_convergence=res -> begin
        # without convergence check, stop after 5000 iterations
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
)

ϵ_opt = result.optimized_controls[1]


# Or, using the QuantumControl API (recommended)

using QuantumControl: ControlProblem, optimize, @optimize_or_load

problem = ControlProblem(
    [traj], tlist,
    prop_method = ExpProp,
    J_T = J_T_sm,
    check_convergence=res -> begin
        ((res.J_T < 1e-3) && (res.converged = true) && (res.message = "J_T < 10⁻³"))
    end,
)

result = optimize(problem; method=GRAPE)

# This dumps the optimization result in `tls_opt.jld2`
result = @optimize_or_load("tls_opt.jld2", problem; method = GRAPE)
```

See the [Tutorial](https://juliaquantumcontrol.github.io/GRAPE.jl/stable/tutorial/) and [Usage section](https://juliaquantumcontrol.github.io/GRAPE.jl/stable/usage/) in the documentation for more details.

## Documentation

The documentation of `GRAPE.jl` is available at <https://juliaquantumcontrol.github.io/GRAPE.jl>.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/JuliaQuantumControl/.github/blob/master/CONTRIBUTING.md#contributing-to-juliaquantumcontrol-packages) and the [organization development notes](https://github.com/JuliaQuantumControl#development).


[`QuantumControl.jl`]: https://github.com/JuliaQuantumControl/QuantumControl.jl#readme
[JuliaQuantumControl]: https://github.com/JuliaQuantumControl

## History

See the [`CHANGELOG.md`](CHANGELOG.md) and the [Release Notes](https://github.com/JuliaQuantumControl/GRAPE.jl/releases).

## License

The source code of this project is licensed under the [MIT License](LICENSE). The documentation is licensed under [Creative Commons (`CC-BY-4.0`)](https://creativecommons.org/licenses/by/4.0/deed.en). License information for all files is [automatically tracked](https://github.com/JuliaQuantumControl/GRAPE.jl/actions/workflows/REUSE.yml) according to [REUSE](https://reuse.software) and can be verified using the [`reuse` tool](https://github.com/fsfe/reuse-tool?tab=readme-ov-file#reuse), e.g., by running `reuse spdx`.
