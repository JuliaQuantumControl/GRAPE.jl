# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: CC0-1.0

using QuantumControl
using QuantumPropagators
using GRAPE
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Pkg
import Optim
import LBFGSB


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
    "Zygote" => "https://fluxml.ai/Zygote.jl/dev/",
    "Optim" => "https://julianlsolvers.github.io/Optim.jl/stable/",
    "QuantumPropagators" => "https://juliaquantumcontrol.github.io/QuantumPropagators.jl/$DEV_OR_STABLE",
    "QuantumGradientGenerators" => "https://juliaquantumcontrol.github.io/QuantumGradientGenerators.jl/$DEV_OR_STABLE",
    "QuantumControl" => "https://juliaquantumcontrol.github.io/QuantumControl.jl/$DEV_OR_STABLE",
    "Krotov" => "https://juliaquantumcontrol.github.io/Krotov.jl/$DEV_OR_STABLE",
    "Examples" => "https://juliaquantumcontrol.github.io/QuantumControlExamples.jl/$DEV_OR_STABLE",
)

fallbacks = ExternalFallbacks(
    "ControlProblem" => "@extref QuantumControl :jl:type:`QuantumControl.ControlProblem`",
    "QuantumControl.ControlProblem" => "@extref QuantumControl :jl:type:`QuantumControl.ControlProblem`",
    "make_chi" => "@extref QuantumControl.make_chi",
    "QuantumControl.QuantumPropagators.Controls.get_controls" => "@extref QuantumPropagators.Controls.get_controls",
    "Trajectory" => "@extref QuantumControl.Trajectory",
    "propagate" => "@extref QuantumPropagators.propagate",
    "init_prop_trajectory" => "@extref QuantumControl.init_prop_trajectory",
    "Functional" => "@extref QuantumControl :std:label:`Functional`",
)

println("Starting makedocs")

bib = CitationBibliography(joinpath(@__DIR__, "..", "paper", "paper.bib"); style=:numeric)

PAGES = [
    "Home" => "index.md",
    "Usage" => "usage.md",
    "Background" => "background.md",
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
        # mathengine=KaTeX(),
        mathengine=MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"],
                ),
            )
        ),
        footer="[$NAME.jl]($GITHUB) v$VERSION docs [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/deed.en). Powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).",
    ),
    pages=PAGES,
    warnonly=true,
)

println("Finished makedocs")

deploydocs(;
    repo="github.com/JuliaQuantumControl/GRAPE.jl",
    devbranch="master",
    push_preview=true
)
