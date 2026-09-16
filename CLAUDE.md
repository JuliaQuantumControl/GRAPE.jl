# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRAPE.jl is a Julia package implementing GRadient Ascent Pulse Engineering (GRAPE) for quantum control optimization. It is part of the JuliaQuantumControl organization and designed to work with the QuantumControl.jl framework.

## Key Architecture

- **Core Module Structure**: The main module is in `src/GRAPE.jl` which includes three main components:
  - `workspace.jl`: Defines the GRAPE workspace containing trajectories, controls, gradients, and optimizer state
  - `result.jl`: Handles optimization results and iteration data
  - `optimize.jl`: Core optimization algorithm implementing the GRAPE method
- **Extensions**: Optional extensions in `ext/` for additional optimizers (Optim.jl, LBFGSB)
- **Dependencies**: Built on QuantumControl.jl, QuantumGradientGenerators.jl, and LBFGSB.jl

## Development Commands

Run `make help` for all targets. The development workflow is documented in the org-wide [CONTRIBUTING.md](https://github.com/JuliaQuantumControl/.github/blob/master/CONTRIBUTING.md) (`../.github/CONTRIBUTING.md` in the development environment).

- `make test`: Run the test suite in the `test` environment (or `julia --project=test -e 'include("test/runtests.jl")'`)
- `make devrepl`: REPL with the `test` environment active and the `docs` environment stacked; run individual test files (`include("test/test_tls_optimization.jl")`), `include("test/runtests.jl")`, or `include("docs/make.jl")` from there
- `make docs`: Build the documentation in the `docs` environment
- `make coverage` / `make htmlcoverage`: Test coverage
- `make codestyle`: Apply JuliaFormatter (version pinned in the `Makefile`) and check `CHANGELOG.md` and `[sources]`
- `make reuse`: Check REUSE compliance
- `make paper`: Compile the JOSS manuscript in `./paper`
- `make clean` / `make distclean`: Remove build/test artifacts

Sibling packages (QuantumControl, QuantumPropagators, Krotov, …) come from their registered releases, or temporarily from a GitHub branch via a URL `[sources]` entry in `test/Project.toml` / `docs/Project.toml`. Never commit a `path` source for a sibling (as written by `../scripts/installorg.jl`). The `test` and `docs` environments reference the package itself via `[sources]` (`{path = ".."}`); this needs Julia ≥ 1.11.

## Testing Framework

The package uses SafeTestsets.jl for isolated test execution. Tests use `QuantumControl.DummyOptimization` (experimental) for dummy control problems and `QuantumControlTestUtils.RandomObjects` for random states and matrices.

## Package Structure

- **Main Source**: `src/` contains the core GRAPE implementation
- **Extensions**: `ext/` for optional optimizer backends
- **Testing**: `test/` with comprehensive test suite using SafeTestsets
- **Documentation**: `docs/` with full API documentation and usage examples

## Development Notes

- Part of the JuliaQuantumControl ecosystem
- Code formatting follows JuliaQuantumControl organization standards

## General Guidelines

* Make sure to only use explicit imports in Julia code, and that there are no imported functions or constants that are not actually used.

* When adding a new dependency to any `Project.toml` file, run `make distclean`, and then `make test/Manifest.toml`, `make docs/Manifest.toml`, etc. to recreate manifest files as necessary.

* Never commit any changes or ask to commit. I will always create git commits manually.
