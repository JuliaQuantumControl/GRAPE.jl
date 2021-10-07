using Pkg # TODO: remove after GRAPE is registered
Pkg.develop(PackageSpec(path = pwd())) # TODO: remove after GRAPE is registered
using GRAPE
using Documenter

# Generate examples
include("generate.jl")

DocMeta.setdocmeta!(GRAPE, :DocTestSetup, :(using GRAPE); recursive = true)

println("Starting makedocs")

makedocs(;
    modules = [GRAPE],
    authors = "Alastair Marshall <alastair@nvision-imaging.com> and contributors",
    repo = "https://github.com/JuliaQuantumControl/GRAPE.jl/blob/{commit}{path}#{line}",
    sitename = "GRAPE.jl",
    format = Documenter.HTML(;
        prettyurls = true,
        canonical = "https://juliaquantumcontrol.github.io/GRAPE.jl",
        assets = String[],
        mathengine = KaTeX(),
    ),
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Examples" => [
            "List of Examples" => "examples/index.md",
            "Example 1 (TLS)" => "examples/simple_state_to_state.md",
        ],
        "API" => "api.md",
    ],
)

println("Finished makedocs")

rm(joinpath(@__DIR__, "build", "examples", ".gitignore"))

deploydocs(;
    repo = "github.com/JuliaQuantumControl/GRAPE.jl",
    devbranch = "master",
)
