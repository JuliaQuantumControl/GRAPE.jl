module GRAPE

include("workspace.jl")
include("result.jl")
include("optimize.jl")
include("backend_lbfgsb.jl")  # other backends are package extensions

end
