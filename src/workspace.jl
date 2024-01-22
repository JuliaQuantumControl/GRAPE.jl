import QuantumControlBase
using QuantumControlBase.QuantumPropagators: init_prop
using QuantumControlBase.QuantumPropagators.Storage: init_storage
using QuantumControlBase.QuantumPropagators.Controls: get_controls, discretize_on_midpoints
using QuantumControlBase: Trajectory, get_control_derivs, init_prop_trajectory
using QuantumGradientGenerators: GradVector, GradGenerator
using ConcreteStructs

# GRAPE workspace (for internal use)
@concrete terse mutable struct GrapeWrk

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

    # search-direction for guess in iterations
    searchdirection::Vector{Float64}

    # Upper bound for every `pulsevals`, +Inf indicates no bound
    upper_bounds::Vector{Float64}

    # Upper bound for every `pulsevals`, -Inf indicates no bound
    lower_bounds::Vector{Float64}

    # the step width in the search direction
    alpha::Float64

    fg_count::Vector{Int64}

    # map of controls to options
    pulse_options

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
    # interleave the pulse values as [ϵ₁(t̃₁), ϵ₂(t̃₁), ..., ϵ₁(t̃₂), ϵ₂(t̃₂), ...]
    # to allow access as reshape(pulsevals, L, :)[l, n] where l is the control
    # index and n is the time index
    pulsevals = convert(
        Vector{Float64},
        reshape(
            transpose(
                hcat([discretize_on_midpoints(control, tlist) for control in controls]...)
            ),
            :
        )
    )
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
            reshape(
                transpose(
                    hcat(
                        [
                            discretize_on_midpoints(control, result.tlist) for
                            control in result.optimized_controls
                        ]...
                    )
                ),
                :
            )
        )
    else
        result = GrapeResult(problem)
    end
    parameters = IdDict(
        # The view-aliasing below ensures that we can mutate `pulsevals` and
        # the updated values are immediately accessible in the propagation
        control => @view pulsevals[l:length(controls):end] for
        (l, control) in enumerate(controls)
    )
    gradient = zeros(length(pulsevals))
    grad_J_T = zeros(length(pulsevals))
    grad_J_a = zeros(length(pulsevals))
    J_parts = zeros(2)
    pulsevals_guess = copy(pulsevals)
    searchdirection = zeros(length(pulsevals))
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
    alpha = 0.0
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
        [zeros(ComplexF64, length(controls), length(tlist) - 1) for _ in trajectories]
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
    GrapeWrk(
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
        searchdirection,
        upper_bounds,
        lower_bounds,
        alpha,
        fg_count,
        pulse_options,
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
