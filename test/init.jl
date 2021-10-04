# init file for "make devrepl"
using Pkg  # TODO: remove after GRAPE is registered
Pkg.develop(PackageSpec(path = pwd()))  # TODO: remove after GRAPE is registered
using Revise
println("""
*******************************************************************************
DEVELOPMENT REPL

Revise is active

Run

    include("test/runtests.jl")

for running the entire test suite. Alternatively, run e.g.

    include("test/generate_example_tests.jl")
    include("test/examples/simple_state_to_state.jl")

for running an individual example test file. Run

    include("docs/make.jl")

to generate the documentation
*******************************************************************************
""")
