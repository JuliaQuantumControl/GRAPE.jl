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

### Testing
```bash
make test                    # Run full test suite
julia --project=test -e 'include("devrepl.jl"); test()'  # Alternative test command
- Test individual files by running them from the test REPL
```

### Documentation
```bash
make docs                    # Build documentation
```

### Development Environment
```bash
make devrepl                # Start interactive REPL with test environment
julia -i --banner=no devrepl.jl  # Alternative way to start dev REPL
```

### Code Formatting
```bash
make codestyle              # Apply JuliaFormatter to entire project
```

### Cleaning
```bash
make clean                  # Clean build/doc/testing artifacts
make distclean             # Restore to clean checkout state
```

## Testing Framework

The package uses SafeTestsets.jl for isolated test execution.

Single test files can be run directly: `julia --project=test -e 'include("test/test_tls_optimization.jl")'`

## Package Structure

- **Main Source**: `src/` contains the core GRAPE implementation
- **Extensions**: `ext/` for optional optimizer backends
- **Testing**: `test/` with comprehensive test suite using SafeTestsets
- **Documentation**: `docs/` with full API documentation and usage examples

## Development Notes

- Part of JuliaQuantumControl ecosystem - may use shared development scripts in `../scripts/`
- This package is designed to work within the JuliaQuantumControl development environment
- Code formatting follows JuliaQuantumControl organization standards
- Tests require the full test environment with additional dependencies
- Uses `devrepl.jl` for unified development environment setup
