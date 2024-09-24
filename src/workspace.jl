import QuantumControl
using QuantumControl.QuantumPropagators.Storage: init_storage
using QuantumControl.QuantumPropagators.Controls: get_controls, discretize_on_midpoints
using QuantumControl: Trajectory, init_prop_trajectory
using QuantumControl.Controls: get_control_derivs
using QuantumGradientGenerators: GradVector, GradGenerator
import LBFGSB

"""GRAPE Workspace.

The workspace is for internal use. However, it is also accessible in a
`callback` function. The callback may use or modify some of the following
attributes:

* `trajectories`: a copy of the trajectories defining the control problem
* `adjoint_trajectories`: The `trajectories` with the adjoint generator
* `kwargs`: The keyword arguments from the [`ControlProblem`](@ref) or the
  call to [`optimize`](@ref).
* `controls`: A tuple of the original controls (probably functions)
* `pulsevals_guess`: The combined vector of pulse values that are the guess in
  the current iteration. Initially, the vector is the concatenation of
  discretizing `controls` to the midpoints of the time grid.
* `pulsevals`: The combined vector of updated pulse values in the current
  iteration.
* `gradient`: The total gradient for the guess in the current iteration
* `grad_J_T`: The *current*  gradient for the final-time part of the
  functional. This is from the last evaluation of the gradient, which may be
  for the optimized pulse (depending on the internal of the optimizer)
* `grad_J_a`: The *current*  gradient for the running cost part of the
  functional.
* `J_parts`: The two-component vector ``[J_T, J_a]``
* `result`: The current result object
* `upper_bounds`: Upper bound for every `pulsevals`; `+Inf` indicates no bound.
* `lower_bounds`: Lower bound for every `pulsevals`; `-Inf` indicates no bound.
* `fg_count`: The total number of evaluations of the functional and evaluations
  of the gradient, as a two-element vector.
* `optimizer`: The backend optimizer object
* `optimizer_state`: The internal state object of the `optimizer` (`nothing` if
  the `optimizer` has no internal state)
* `result`: The current result object
* `tau_grads`: The gradients ∂τₖ/ϵₗ(tₙ)
* `fw_storage`: The storage of states for the forward propagation
* `fw_propagators`: The propagators used for the forward propagation
* `bw_grad_propagators`: The propagators used for the backward propagation of
  [`QuantumGradientGenerators.GradVector`](@ref) states
  (`gradient_method=:gradgen` only)
* `bw_propagators`: The propagators used for the backward propagation
  (`gradient_method=:taylor` only)
* `use_threads`: Flag indicating whether the propagations are performed in
  parallel.

In addition, the following methods provide safer (non-mutating) access to
information in the workspace

* [`step_width`](@ref)
* [`search_direction`](@ref)
* [`norm_search`](@ref)
* [`gradient`](@ref)
* [`pulse_update`](@ref)
"""
mutable struct GrapeWrk{O}

    trajectories
    adjoint_trajectories
    # trajectories for bw-prop of gradients
    grad_trajectories
    kwargs
    controls
    pulsevals_guess::Vector{Float64}
    pulsevals::Vector{Float64}
    gradient::Vector{Float64}
    grad_J_T::Vector{Float64}
    grad_J_a::Vector{Float64}
    J_parts::Vector{Float64}
    upper_bounds::Vector{Float64}
    lower_bounds::Vector{Float64}
    fg_count::Vector{Int64}
    optimizer::O
    optimizer_state
    J_T_takes_tau::Bool  # Does J_T have a tau keyword arg?
    chi_takes_tau::Bool # Does chi have a tau keyword arg?
    result

    #################################
    # Per trajectory:

    # χ(T), for easier debugging in a callback
    chi_states

    tau_grads::Vector{Matrix{ComplexF64}}
    fw_storage
    fw_prop_kwargs::Vector{Dict{Symbol,Any}}
    bw_grad_prop_kwargs::Vector{Dict{Symbol,Any}}
    bw_prop_kwargs::Vector{Dict{Symbol,Any}}

    # for normal forward propagation
    fw_propagators

    # for gradient backward propagation
    # gradient_method=:gradgen only
    bw_grad_propagators

    # for normal backward propagation
    # gradient_method=:taylor only
    bw_propagators

    # evaluated Hₖ for a particular point in time
    # gradient_method=:taylor only
    taylor_genops

    # derivatives ∂Hₖ/∂ϵₗ(t)
    # gradient_method=:taylor only
    control_derivs

    # 5 temporary states for each trajectory and each control, for evaluating
    # gradients via Taylor expansions
    # gradient_method=:taylor only
    taylor_grad_states

    use_threads::Bool

end


function GrapeWrk(problem::QuantumControl.ControlProblem; verbose=false)
    use_threads = get(problem.kwargs, :use_threads, false)
    gradient_method = get(problem.kwargs, :gradient_method, :gradgen)
    trajectories = [traj for traj in problem.trajectories]
    N = length(trajectories)
    adjoint_trajectories = [adjoint(traj) for traj in problem.trajectories]
    controls = get_controls(trajectories)
    if length(controls) == 0
        error("no controls in trajectories: cannot optimize")
    end
    tlist = problem.tlist
    N_T = length(tlist) - 1
    # Concatenate pulse values. For `N_T = length(tlist) - 1` time intervals,
    # `pulsesvals[(l-1)*N_T + n]` is the value for the l'th control and time
    # index n
    pulsevals = vcat([discretize_on_midpoints(control, tlist) for control in controls]...)
    kwargs = Dict(problem.kwargs)  # creates a shallow copy; ok to modify
    default_pulse_options = IdDict()  # not used
    pulse_options = get(kwargs, :pulse_options, default_pulse_options)
    fg_count = zeros(Int64, 2)
    if haskey(kwargs, :continue_from)
        @info "Continuing previous optimization"
        result = kwargs[:continue_from]
        if !(result isa GrapeResult)
            # account for continuing from a different optimization method
            result = convert(GrapeResult, result)
        end
        result.iter_stop = get(kwargs, :iter_stop, 5000)
        result.converged = false
        result.start_local_time = now()
        result.message = "in progress"
        pulsevals = convert(
            Vector{Float64},
            vcat(
                [
                    discretize_on_midpoints(control, result.tlist) for
                    control in result.optimized_controls
                ]...
            )
        )
    else
        result = GrapeResult(problem)
    end
    parameters = IdDict(
        # The view-aliasing below ensures that we can mutate `pulsevals` and
        # the updated values are immediately accessible in the propagation
        control => @view pulsevals[(l-1)*N_T+1:l*N_T] for
        (l, control) in enumerate(controls)
    )
    gradient = zeros(length(pulsevals))
    grad_J_T = zeros(length(pulsevals))
    grad_J_a = zeros(length(pulsevals))
    J_parts = zeros(2)
    pulsevals_guess = copy(pulsevals)
    upper_bounds = fill(get(kwargs, :upper_bound, Inf), length(pulsevals))
    lower_bounds = fill(get(kwargs, :lower_bound, -Inf), length(pulsevals))
    for (l, control) in enumerate(controls)
        options = get(pulse_options, control, Dict())
        if haskey(options, :upper_bounds)
            ub = @view upper_bounds[l:length(controls):end]
            ub .= options[:upper_bounds]
        end
        if haskey(options, :lower_bounds)
            lb = @view lower_bounds[l:length(controls):end]
            lb .= options[:lower_bounds]
        end
    end
    fw_storage = [init_storage(traj.initial_state, tlist) for traj in trajectories]
    kwargs[:piecewise] = true  # only accept piecewise propagators
    _prefixes = ["prop_", "fw_prop_"]
    fw_prop_kwargs = [Dict{Symbol,Any}() for _ = 1:N]
    bw_grad_prop_kwargs = [Dict{Symbol,Any}() for _ = 1:N]
    bw_prop_kwargs = [Dict{Symbol,Any}() for _ = 1:N]
    fw_propagators = [
        init_prop_trajectory(
            traj,
            tlist;
            verbose,
            _msg="Initializing fw-prop of trajectory $k",
            _prefixes,
            _filter_kwargs=true,
            _kwargs_dict=fw_prop_kwargs[k],
            fw_prop_parameters=parameters,  # will filter to `parameters`
            kwargs...
        ) for (k, traj) in enumerate(trajectories)
    ]
    chi_states = [zero(traj.initial_state) for traj in trajectories]
    tau_grads::Vector{Matrix{ComplexF64}} =
        [zeros(ComplexF64, length(tlist) - 1, length(controls)) for _ = 1:N]
    if gradient_method == :gradgen
        grad_trajectories = [
            begin
                χ̃ₖ = GradVector(chi_states[k], length(controls))
                G̃ₖ = GradGenerator(traj.generator)
                Trajectory(χ̃ₖ, G̃ₖ; getfield(traj, :kwargs)...)
            end for (k, traj) in enumerate(adjoint_trajectories)
        ]
        _prefixes = ["prop_", "bw_prop_", "grad_prop_"]
        bw_grad_propagators = [
            init_prop_trajectory(
                traj,
                tlist;
                verbose,
                _msg="Initializing bw-gradient-prop of trajectory $k",
                _prefixes,
                _filter_kwargs=true,
                _kwargs_dict=bw_grad_prop_kwargs[k],
                grad_prop_backward=true,  # will filter to `backward=true`
                grad_prop_parameters=parameters,  # will filter to `parameters`
                kwargs...
            ) for (k, traj) in enumerate(grad_trajectories)
        ]
        bw_propagators = []
        taylor_genops = []
        control_derivs = []
        taylor_grad_states = []
    elseif gradient_method == :taylor
        grad_trajectories = []
        bw_grad_propagators = []
        _prefixes = ["prop_", "bw_prop_"]
        bw_propagators = [
            init_prop_trajectory(
                traj,
                tlist;
                verbose,
                _msg="Initializing bw-prop of trajectory $k",
                _prefixes,
                _filter_kwargs=true,
                _kwargs_dict=bw_prop_kwargs[k],
                bw_prop_backward=true,  # will filter to `backward=true`
                bw_prop_parameters=parameters,  # will filter to `parameters`
                kwargs...
            ) for (k, traj) in enumerate(adjoint_trajectories)
        ]
        taylor_genops =
            [evaluate(traj.generator, tlist, 1) for traj in adjoint_trajectories]
        control_derivs =
            [get_control_derivs(traj.generator, controls) for traj in trajectories]
        taylor_grad_states = [
            Tuple(similar(trajectories[k].initial_state) for _ = 1:5) for
            l in eachindex(controls), k in eachindex(trajectories)
        ]
    else
        error("Invalid gradient_method=$(repr(gradient_method)) ∉ (:gradgen, :taylor)")
    end
    optimizer = get_optimizer(length(pulsevals); kwargs...)
    optimizer_state = nothing  # set in run_optimizer, if applicable
    O = typeof(optimizer)
    J_T_takes_tau = false
    if haskey(kwargs, :J_T)
        J_T = kwargs[:J_T]
    else
        msg = "`optimize` for `method=GRAPE` must be passed the functional `J_T`."
        throw(ArgumentError(msg))
    end
    J_T_takes_tau =
        hasmethod(J_T, Tuple{typeof(result.states),typeof(trajectories)}, (:tau,))
    if !haskey(kwargs, :chi)
        kwargs[:chi] = make_chi(J_T, trajectories)
    end
    chi = kwargs[:chi]
    chi_takes_tau =
        hasmethod(chi, Tuple{typeof(result.states),typeof(trajectories)}, (:tau,))
    GrapeWrk{O}(
        trajectories,
        adjoint_trajectories,
        grad_trajectories,
        kwargs,
        controls,
        pulsevals_guess,
        pulsevals,
        gradient,
        grad_J_T,
        grad_J_a,
        J_parts,
        upper_bounds,
        lower_bounds,
        fg_count,
        optimizer,
        optimizer_state,
        J_T_takes_tau,
        chi_takes_tau,
        result,
        chi_states,
        tau_grads,
        fw_storage,
        fw_prop_kwargs,
        bw_grad_prop_kwargs,
        bw_prop_kwargs,
        fw_propagators,
        bw_grad_propagators,
        bw_propagators,
        taylor_genops,
        control_derivs,
        taylor_grad_states,
        use_threads
    )
end


function get_optimizer(n; kwargs...)
    m = 10 # TODO: kwarg for number of limited memory corrections
    optimizer = get(kwargs, :optimizer, LBFGSB.L_BFGS_B(n, m))
    return optimizer
end


"""The step width used in the current iteration.

```julia
α = step_width(wrk)
```

returns the scalar `α` so that `pulse_update(wrk) = α * search_direction(wrk)`,
see [`pulse_update`](@ref) and [`search_direction`](@ref) for the iteration
described by the current [`GrapeWrk`](@ref) (for the state of `wrk` as available
in the `callback` of the current iteration.
"""
function step_width(wrk)
    u = pulse_update(wrk)
    s = search_direction(wrk)
    ϕ = vec_angle(u, s)
    if abs(ϕ) > 1e-10
        @warn "pulse_update is not parallel to search_direction (angle $(ϕ)rad)"
    end
    return norm(u) / norm(s)

end


"""The search direction used in the current iteration.

```julia
s = search_direction(wrk)
```

returns the vector describing the search direction used in the current
iteration. This should be proportional to [`pulse_update`](@ref) with the
proportionality factor [`step_width`](@ref).
"""
search_direction(wrk) = -gradient(wrk; which=:initial)  # assumed fallback


"""The norm of the search direction vector in the current iteration.

```julia
norm_search(wrk)
```

returns `norm(search_direction(wrk))`.
"""
function norm_search(wrk)
    s = search_direction(wrk)
    r = 0.0
    for sᵢ in s
        r += sᵢ^2
    end
    return sqrt(r)
end


"""The gradient in the current iteration.

```julia
g = gradient(wrk; which=:initial)
```

returns the gradient associated with the guess pulse of the current iteration.
Up to quasi-Newton corrections, the negative gradient determines the
[`search_direction`](@ref) for the [`pulse_update`](@ref).

```julia
g = gradient(wrk; which=:final)
```

returns the gradient associated with the optimized pulse of the current
iteration.
"""
function gradient(wrk; which=:initial)
    if which == :initial
        return wrk.gradient
    elseif which == :final
        λₐ = get(wrk.kwargs, :lambda_a, 1.0)
        G = copy(wrk.grad_J_T)
        axpy!(λₐ, wrk.grad_J_a, G)
        return G
    else
        throw(ArgumentError("`which` must be :initial or :final, not $(repr(which))"))
    end
end


"""The vector of pulse update values for the current iteration.

```julia
Δu = pulse_update(wrk)
```

returns a vector containing the different between the optimized pulse values
and the guess pulse values of the current iteration. This should be
proportional to [`search_direction`](@ref) with the
proportionality factor [`step_width`](@ref).
"""
pulse_update(wrk) = wrk.pulsevals - wrk.pulsevals_guess


"""The angle between two vectors.

```
ϕ = vec_angle(v1, v2; unit=:rad)
```

returns the angle between two vectors in radians (or degrees, with
`unit=:degree`).
"""
function vec_angle(
    vec1::P,
    vec2::P;
    unit=:rad
) where {P<:Union{NTuple{N,T},AbstractVector{T}}} where {N,T}
    # `vec_angle` function adapted from AngleBetweenVectors.jl
    # by Jeffrey Sarnoff, licensed under the terms of the MIT license
    unitvec1 = unitize(vec1)
    unitvec2 = unitize(vec2)
    y = unitvec1 .- unitvec2
    x = unitvec1 .+ unitvec2
    a = 2 * atan(norm(y), norm(x))
    if signbit(a) || signbit(float(T)(pi) - a)
        a = signbit(a) ? zero(T) : float(T)(pi)
    end
    if unit == :degree
        a = a * 180 / π
    elseif unit != :rad
        msg = "`unit` must be :rad or :degree, not $(repr(unit))"
        throw(ArgumentError(msg))
    end
    return a
end

@inline unitize(v) = v ./ norm(v)
