using GRAPE
using Pkg
using Documenter
using Plots

gr()
ENV["GKSwstype"] = "100"
ENV["GRAPE_LINESEARCH_ANALYSIS_VERBOSE"] = "0"

include(joinpath(@__DIR__, "download_dumps.jl"))

# Generate examples
include("generate.jl")

DocMeta.setdocmeta!(GRAPE, :DocTestSetup, :(using GRAPE); recursive=true)

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/GRAPE.jl"

println("Starting makedocs")

makedocs(;
    modules=[GRAPE],
    authors=AUTHORS,
    repo="$GITHUB/blob/{commit}{path}#{line}",
    sitename="GRAPE.jl",
    format=Documenter.HTML(;
        prettyurls = true,
        canonical  = "https://juliaquantumcontrol.github.io/GRAPE.jl",
        assets     = String[],
        mathengine = KaTeX(),
        footer     = "[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Examples" => [
            "List of Examples" => "examples/index.md",
            "Example 1 (TLS)" => "examples/simple_state_to_state.md",
        ],
        "API" => "api.md",
        "History" => "history.md",
    ]
)

println("Finished makedocs")

rm(joinpath(@__DIR__, "build", "examples", ".gitignore"))

deploydocs(; repo="github.com/JuliaQuantumControl/GRAPE.jl", devbranch="master")
