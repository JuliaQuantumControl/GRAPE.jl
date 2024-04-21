module GRAPEKrotovExt

using GRAPE: GrapeResult
using Krotov: KrotovResult

function Base.convert(::Type{GrapeResult}, result::KrotovResult{STST}) where {STST}
    tau_vals = zeros(ComplexF64, length(result.states))
    # TODO: if Krotov were to properly calculate τ values, we could just copy
    # them here
    return GrapeResult{STST}(
        result.tlist,
        result.iter_start,
        result.iter_stop,
        result.iter,
        result.secs,
        tau_vals,
        result.J_T,
        result.J_T_prev,
        result.guess_controls,
        result.optimized_controls,
        result.states,
        result.start_local_time,
        result.end_local_time,
        result.records,
        result.converged,
        0, # f_calls
        0, # fg_calls
        result.message,
    )
end


function Base.convert(::Type{KrotovResult}, result::GrapeResult{STST}) where {STST}
    return KrotovResult{STST}(
        result.tlist,
        result.iter_start,
        result.iter_stop,
        result.iter,
        result.secs,
        result.tau_vals,
        result.J_T,
        result.J_T_prev,
        result.guess_controls,
        result.optimized_controls,
        result.states,
        result.start_local_time,
        result.end_local_time,
        result.records,
        result.converged,
        result.message,
    )
end

end
