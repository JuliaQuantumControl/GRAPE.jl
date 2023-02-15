# init file for "make devrepl"
using Revise
using Plots
unicodeplots()
using JuliaFormatter
using QuantumControlTestUtils: test, show_coverage, generate_coverage_html
using LiveServer: LiveServer, serve, servedocs as _servedocs
using Term
include(joinpath(@__DIR__, "clean.jl"))

servedocs(; kwargs...) = _servedocs(; skip_dirs=["docs/src/examples"], kwargs...)

REPL_MESSAGE = """
*******************************************************************************
DEVELOPMENT REPL

Revise, JuliaFormatter, LiveServer, Plots with unicode backend are active.

* `help()` – Show this message
* `include("test/runtests.jl")` – Run the entire test suite
* `include("test/generate_example_tests.jl")` – Convert all examples to tests
* `include("test/examples/simple_state_to_state.jl")` –
  Run an individual example test (after converting examples)!
* `test()` – Run the entire test suite in a subprocess with coverage
* `show_coverage()` – Print a tabular overview of coverage data
* `generate_coverage_html()` – Generate an HTML coverage report
* `include("docs/make.jl")` – Generate the documentation
* `format(".")` – Apply code formatting to all files
* `servedocs([port=8000, verbose=false])` –
  Build and serve the documentation. Automatically recompile and redisplay on
  changes
* `clean()` – Clean up build/doc/testing artifacts
* `distclean()` – Restore to a clean checkout state
*******************************************************************************
"""

"""Show help"""
help() = println(REPL_MESSAGE)
