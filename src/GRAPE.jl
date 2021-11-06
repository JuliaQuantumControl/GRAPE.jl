module GRAPE

include("optimize.jl")

export GrapeWrk, optimize_grape

include("backend_optim.jl")
include("backend_lbfgsb.jl")


import QuantumControlBase: optimize

"""
```julia
opt_result = optimize(problem; method=:grape, kwargs...)
```

optimizes `problem` using GRadident Ascent Pulse Engineering (GRAPE), see
[`GRAPE.optimize_grape`](@ref).
"""
optimize(problem, method::Val{:grape}; kwargs...) = GRAPE.optimize_grape(problem, kwargs...)


end
