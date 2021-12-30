module GRAPE

include("workspace.jl")
include("result.jl")
include("optimize.jl")
include("backend_optim.jl")
include("backend_lbfgsb.jl")


import QuantumControlBase

"""
```julia
opt_result = optimize(problem; method=:GRAPE, kwargs...)
```

optimizes [`problem`](@ref QuantumControlBase.ControlProblem)
using GRadident Ascent Pulse Engineering (GRAPE), see
[`GRAPE.optimize_grape`](@ref).
"""
QuantumControlBase.optimize(problem, method::Val{:GRAPE}; kwargs...) = optimize_grape(problem; kwargs...)
QuantumControlBase.optimize(problem, method::Val{:grape}; kwargs...) = optimize_grape(problem; kwargs...)


end
