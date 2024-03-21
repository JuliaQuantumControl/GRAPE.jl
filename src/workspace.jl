import QuantumControlBase
using QuantumControlBase.QuantumPropagators.Storage: init_storage
using QuantumControlBase.QuantumPropagators.Controls: get_controls, discretize_on_midpoints
using QuantumControlBase: Trajectory, get_control_derivs, init_prop_trajectory
using QuantumGradientGenerators: GradVector, GradGenerator
import LBFGSB

"""Grape Workspace.

# Methods

* [`step_width`](@ref)
* [`search_direction`](@ref)
* [`gradient`](@ref)
* [`pulse_update`](@ref)
"""
mutable struct GrapeWrk{O}

    # a copy of the trajectories
    trajectories

    # the adjoint trajectories, containing the adjoint generators for the
    # backward propagation
    adjoint_trajectories

    # trajectories for bw-prop of gradients
    grad_trajectories

    # The kwargs from the control problem
    kwargs

    # Tuple of the original controls (probably functions)
    controls

    pulsevals_guess::Vector{Float64}

    pulsevals::Vector{Float64}

    # total gradient for guess in iterations
    gradient::Vector{Float64}

    # storage for current final time gradient
    grad_J_T::Vector{Float64}

    # storage for current running cost gradient
    grad_J_a::Vector{Float64}

    # two-component vector [J_T, J_a]
    J_parts::Vector{Float64}

    # Upper bound for every `pulsevals`, +Inf indicates no bound
    upper_bounds::Vector{Float64}

    # Upper bound for every `pulsevals`, -Inf indicates no bound
    lower_bounds::Vector{Float64}

    fg_count::Vector{Int64}

    # map of controls to options
    pulse_options

    # The optimizer
    optimizer::O

    # Internal optimizer state (`nothing` if `optimizer` has not state)
    optimizer_state

    result

    #################################
    # scratch objects, per trajectory:

    # backward-propagated states
    chi_states

    # gradients ∂τₖ/ϵₗ(tₙ)
    tau_grads::Vector{Matrix{ComplexF64}}

    # backward storage array
    fw_storage

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

function GrapeWrk(problem::QuantumControlBase.ControlProblem; verbose=false)
    use_threads = get(problem.kwargs, :use_threads, false)
    gradient_method = get(problem.kwargs, :gradient_method, :gradgen)
    trajectories = [traj for traj in problem.trajectories]
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
    dummy_vals = IdDict(control => 1.0 for (i, control) in enumerate(controls))
    fw_storage = [init_storage(traj.initial_state, tlist) for traj in trajectories]
    kwargs[:piecewise] = true  # only accept piecewise propagators
    _prefixes = ["prop_", "fw_prop_"]
    fw_propagators = [
        init_prop_trajectory(
            traj,
            tlist;
            verbose,
            _msg="Initializing fw-prop of trajectory $k",
            _prefixes,
            _filter_kwargs=true,
            fw_prop_parameters=parameters,  # will filter to `parameters`
            kwargs...
        ) for (k, traj) in enumerate(trajectories)
    ]
    chi_states = [similar(traj.initial_state) for traj in trajectories]
    tau_grads::Vector{Matrix{ComplexF64}} =
        [zeros(ComplexF64, length(tlist) - 1, length(controls)) for _ in trajectories]
    if gradient_method == :gradgen
        grad_trajectories = [
            begin
                χ̃ₖ = GradVector(chi_states[k], length(controls))
                G̃ₖ = GradGenerator(traj.generator)
                Trajectory(χ̃ₖ, G̃ₖ, getfield(traj, :kwargs)...)
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
            l = 1:length(controls), k = 1:length(trajectories)
        ]
    else
        error("Invalid gradient_method=$(repr(gradient_method)) ∉ (:gradgen, :taylor)")
    end
    optimizer = get_optimizer(length(pulsevals); kwargs...)
    optimizer_state = nothing  # set in run_optimizer, if applicable
    O = typeof(optimizer)
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
        pulse_options,
        optimizer,
        optimizer_state,
        result,
        chi_states,
        tau_grads,
        fw_storage,
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
desribed by the current [`GrapeWrk`](@ref) (for the state of `wrk` as available
in the `info_hook` of the current iteration.
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

returns a vector conntaining the different between the optimized pulse values
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
        throw(ValueError("`unit` must be :rad or :degree, not $(repr(unit))"))
    end
    return a
end

@inline unitize(v) = v ./ norm(v)
