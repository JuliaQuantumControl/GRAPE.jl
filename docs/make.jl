using QuantumControlBase
using QuantumPropagators
using GRAPE
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Pkg
using Plots
import Optim
import LBFGSB

gr()
ENV["GKSwstype"] = "100"


PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/GRAPE.jl"

DEV_OR_STABLE = "stable/"
if endswith(VERSION, "dev")
    DEV_OR_STABLE = "dev/"
end

links = InterLinks(
    "TimerOutputs" => (
        "https://github.com/KristofferC/TimerOutputs.jl",
        joinpath(@__DIR__, "src", "inventories", "TimerOutputs.toml")
    ),
    "QuantumControlBase" => "https://juliaquantumcontrol.github.io/QuantumControlBase.jl/$DEV_OR_STABLE",
    "QuantumPropagators" => "https://juliaquantumcontrol.github.io/QuantumPropagators.jl/$DEV_OR_STABLE",
    "QuantumGradientGenerators" => "https://juliaquantumcontrol.github.io/QuantumGradientGenerators.jl/$DEV_OR_STABLE",
    "QuantumControl" => "https://juliaquantumcontrol.github.io/QuantumControl.jl/$DEV_OR_STABLE",
    "GRAPE" => "https://juliaquantumcontrol.github.io/GRAPE.jl/$DEV_OR_STABLE",
    "Examples" => "https://juliaquantumcontrol.github.io/QuantumControlExamples.jl/$DEV_OR_STABLE",
)

fallbacks = ExternalFallbacks(
    "ControlProblem" => "@extref QuantumControl :jl:type:`QuantumControlBase.ControlProblem`",
    "QuantumControlBase.ControlProblem" => "@extref QuantumControl :jl:type:`QuantumControlBase.ControlProblem`",
    "make_chi" => "@extref QuantumControlBase.make_chi",
    "QuantumControlBase.QuantumPropagators.Controls.get_controls" => "@extref QuantumPropagators.Controls.get_controls",
    "Trajectory" => "@extref QuantumControlBase.Trajectory",
    "propagate" => "@extref QuantumPropagators.propagate",
    "init_prop_trajectory" => "@extref QuantumControlBase.init_prop_trajectory"
)

println("Starting makedocs")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

PAGES = [
    "Home" => "index.md",
    "Overview" => "overview.md",
    "Examples" => "examples.md",
    "API" => "api.md",
    "References" => "references.md",
]

makedocs(;
    plugins=[bib, links, fallbacks],
    modules=[GRAPE],
    authors=AUTHORS,
    sitename="GRAPE.jl",
    doctest=false,  # we have no doctests (but trying to run them is slow)
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://juliaquantumcontrol.github.io/GRAPE.jl",
        assets=[
            "assets/citations.css",
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.css"
            ),
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.js"
            ),
        ],
        mathengine=KaTeX(),
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).",
    ),
    pages=PAGES,
    warnonly=true,
)

println("Finished makedocs")

deploydocs(; repo="github.com/JuliaQuantumControl/GRAPE.jl", devbranch="master")
