# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module GRAPE

include("workspace.jl")
include("result.jl")
include("optimize.jl")
include("docstring.jl")
include(joinpath("..", "ext", "GRAPELBFGSBExt.jl"))

using SciMLPublic: @public
@public optimize, GrapeResult

using QuantumControl: Trajectory, set_default_ad_framework
@public Trajectory, set_default_ad_framework

end
