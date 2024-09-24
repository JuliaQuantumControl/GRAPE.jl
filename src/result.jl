using QuantumControl.QuantumPropagators.Controls: get_controls, discretize
using QuantumControl: AbstractOptimizationResult
using Printf
using Dates

"""Result object returned by [`optimize_grape`](@ref).

# Attributes

The attributes of a `GrapeResult` object include

* `iter`:  The number of the current iteration
* `J_T`: The value of the final-time functional in the current iteration
* `J_T_prev`: The value of the final-time functional in the previous iteration
* `J_a`: The value of the running cost ``J_a`` in the current iteration
  (excluding ``λ_a``)
* `J_a_prev`: The value of ``J_a`` in the previous iteration
* `tlist`: The time grid on which the control are discetized.
* `guess_controls`: A vector of the original control fields (each field
  discretized to the points of `tlist`)
* optimized_controls: A vector of the optimized control fields in the current
  iterations
* records: A vector of tuples with values returned by a `callback` routine
  passed to [`optimize`](@ref)
* converged: A boolean flag on whether the optimization is converged. This
  may be set to `true` by a `check_convergence` function.
* message: A message string to explain the reason for convergence. This may be
  set by a `check_convergence` function.

All of the above attributes may be referenced in a `check_convergence` function
passed to [`optimize(problem; method=GRAPE)`](@ref QuantumControl.optimize(::ControlProblem, ::Val{:GRAPE}))
"""
mutable struct GrapeResult{STST} <: AbstractOptimizationResult
    tlist::Vector{Float64}
    iter_start::Int64  # the starting iteration number
    iter_stop::Int64 # the maximum iteration number
    iter::Int64  # the current iteration number
    secs::Float64  # seconds that the last iteration took
    tau_vals::Vector{ComplexF64}
    J_T::Float64  # the current value of the final-time functional J_T
    J_T_prev::Float64  # previous value of J_T
    J_a::Float64  # the current value of the functional J_a (excluding λ_a)
    J_a_prev::Float64  # previous value of J_a
    guess_controls::Vector{Vector{Float64}}
    optimized_controls::Vector{Vector{Float64}}
    states::Vector{STST}
    start_local_time::DateTime
    end_local_time::DateTime
    records::Vector{Tuple}  # storage for callback to write data into at each iteration
    converged::Bool
    f_calls::Int64
    fg_calls::Int64
    message::String

end

function GrapeResult(problem)
    tlist = problem.tlist
    controls = get_controls(problem.trajectories)
    iter_start = get(problem.kwargs, :iter_start, 0)
    iter_stop = get(problem.kwargs, :iter_stop, 5000)
    iter = iter_start
    secs = 0
    tau_vals = zeros(ComplexF64, length(problem.trajectories))
    guess_controls = [discretize(control, tlist) for control in controls]
    J_T = 0.0
    J_T_prev = 0.0
    J_a = 0.0
    J_a_prev = 0.0
    optimized_controls = [copy(guess) for guess in guess_controls]
    states = [similar(traj.initial_state) for traj in problem.trajectories]
    start_local_time = now()
    end_local_time = now()
    records = Vector{Tuple}()
    converged = false
    message = "in progress"
    f_calls = 0
    fg_calls = 0
    GrapeResult{eltype(states)}(
        tlist,
        iter_start,
        iter_stop,
        iter,
        secs,
        tau_vals,
        J_T,
        J_T_prev,
        J_a,
        J_a_prev,
        guess_controls,
        optimized_controls,
        states,
        start_local_time,
        end_local_time,
        records,
        converged,
        f_calls,
        fg_calls,
        message
    )
end


Base.show(io::IO, r::GrapeResult) = print(io, "GrapeResult<$(r.message)>")
Base.show(io::IO, ::MIME"text/plain", r::GrapeResult) = print(
    io,
    """
GRAPE Optimization Result
-------------------------
- Started at $(r.start_local_time)
- Number of trajectories: $(length(r.states))
- Number of iterations: $(max(r.iter - r.iter_start, 0))
- Number of pure func evals: $(r.f_calls)
- Number of func/grad evals: $(r.fg_calls)
- Value of functional: $(@sprintf("%.5e", r.J_T))
- Reason for termination: $(r.message)
- Ended at $(r.end_local_time) ($(Dates.canonicalize(Dates.CompoundPeriod(r.end_local_time - r.start_local_time))))
"""
)


function Base.convert(::Type{GrapeResult}, result::AbstractOptimizationResult)
    defaults =
        Dict{Symbol,Any}(:f_calls => 0, :fg_calls => 0, :J_a => 0.0, :J_a_prev => 0.0)
    return convert(GrapeResult, result, defaults)
end
