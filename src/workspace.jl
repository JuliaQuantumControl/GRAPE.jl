import QuantumControlBase
using QuantumControlBase: getcontrols, getcontrolderivs, discretize_on_midpoints, evalcontrols
using QuantumControlBase: GradVector, TimeDependentGradGenerator
using QuantumPropagators: init_storage, initpropwrk
using ConcreteStructs

# GRAPE workspace (for internal use)
@concrete terse struct GrapeWrk

    # a copy of the objectives
    objectives

    # the adjoint objectives, containing the adjoint generators for the
    # backward propagation
    adjoint_objectives

    # The kwargs from the control problem
    kwargs

    # Tuple of the original controls (probably functions)
    controls

    # TODO: pulsevals0 and pulsevals1
    pulsevals :: Vector{Float64}

    gradient :: Vector{Float64}  # gradient for guess in iterations

    searchdirection :: Vector{Float64}  # search-direction for guess in iterations

    # Result object
    result

    #################################
    # scratch objects, per objective:

    # backward-propagated states
    # note: storage for fw-propagated states is in result.states
    bw_states

    # forward-propagated states (functional evaluation only)
    fw_states

    # foward-propagated grad-vectors
    fw_grad_states

    # gradients ∂τₖ/ϵₗ(tₙ)
    tau_grads :: Vector{Matrix{ComplexF64}}

    # dynamical generator (normal propagation) at a particular point in time
    G

    # dynamica generator for grad-propagation, time-dependent
    TDgradG

    # dynamical generator for grad-propagation at a particular point in time
    gradG

    control_derivs :: Vector{Vector{Union{Function, Nothing}}}

    vals_dict

    bw_storage # backward storage array (per objective)

    fw_prop_wrk # for normal forward propagation

    bw_prop_wrk # for normal propagation

    grad_prop_wrk # for gradient forward propagation

    use_threads :: Bool

end

function GrapeWrk(problem::QuantumControlBase.ControlProblem)
    use_threads = get(problem.kwargs, :use_threads, false)
    objectives = [obj for obj in problem.objectives]
    adjoint_objectives = [adjoint(obj) for obj in problem.objectives]
    controls = getcontrols(objectives)
    control_derivs = [
        getcontrolderivs(obj.generator, controls) for obj in objectives
    ]
    tlist = problem.tlist
    # interleave the pulse values as [ϵ₁(t̃₁), ϵ₂(t̃₁), ..., ϵ₁(t̃₂), ϵ₂(t̃₂), ...]
    # to allow access as reshape(pulsevals0, L :)[l, n] where l is the control
    # index and n is the time index
    pulsevals = convert(
        Vector{Float64},
        reshape(transpose(hcat(
            [discretize_on_midpoints(control, tlist)
            for control in controls]...
        )), :)
    )
    kwargs = Dict(problem.kwargs)
    pulse_options = problem.pulse_options
    if haskey(kwargs, :continue_from)
        @info "Continuing previous optimization"
        result = kwargs[:continue_from]
        if !(result isa GrapeResult)
            # account for continuing from a different optimization method
            result = convert(GrapeResult, result)
        end
        result.iter_stop = get(problem.kwargs, :iter_stop, 5000)
        result.converged = false
        result.start_local_time = now()
        result.message = "in progress"
        pulsevals = convert(
            Vector{Float64},
            reshape(transpose(hcat(
                [discretize_on_midpoints(control, wrk.result.tlist)
                for control in result.optimized_controls]...
            )), :)
        )
    else
        result = GrapeResult(problem)
    end
    gradient = zeros(length(pulsevals))
    searchdirection = zeros(length(pulsevals))
    bw_states = [similar(obj.initial_state) for obj in objectives]
    fw_states = [similar(obj.initial_state) for obj in objectives]
    fw_grad_states = [
        GradVector(obj.initial_state, length(controls))
        for obj in objectives
    ]
    dummy_vals = IdDict(control => 1.0 for (i, control) in enumerate(controls))
    G = [evalcontrols(obj.generator, dummy_vals) for obj in objectives]
    vals_dict = [copy(dummy_vals) for _ in objectives]
    bw_storage = [init_storage(obj.initial_state, tlist) for obj in objectives]
    fw_prop_method = [
        Val(
            QuantumControlBase.get_objective_prop_method(
                obj, :fw_prop_method, :prop_method; problem.kwargs...
            )
        )
        for obj in objectives
    ]
    bw_prop_method = [
        Val(
            QuantumControlBase.get_objective_prop_method(
                obj, :bw_prop_method, :prop_method; problem.kwargs...
            )
        )
        for obj in objectives
    ]
    grad_prop_method = [
        Val(
            QuantumControlBase.get_objective_prop_method(
                obj, :grad_prop_method, :fw_prop_method, :prop_method;
                problem.kwargs...
            )
        )
        for obj in objectives
    ]

    fw_prop_wrk = [
        QuantumControlBase.initobjpropwrk(obj, tlist, fw_prop_method[k];
                                          initial_state=obj.initial_state,
                                          kwargs...)
        for (k, obj) in enumerate(objectives)
    ]
    bw_prop_wrk = [
        QuantumControlBase.initobjpropwrk(obj, tlist, bw_prop_method[k];
                                          initial_state=obj.initial_state,
                                          kwargs...)
        for (k, obj) in enumerate(objectives)
    ]
    TDgradG = [TimeDependentGradGenerator(obj.generator) for obj in objectives]
    gradG = [evalcontrols(G̃_of_t, dummy_vals) for G̃_of_t ∈ TDgradG]
    grad_prop_wrk = [
        initpropwrk(Ψ̃, tlist, grad_prop_method[k], gradG[k]; kwargs...)
        for (k, Ψ̃) in enumerate(fw_grad_states)
    ]
    tau_grads :: Vector{Matrix{ComplexF64}} = [
        zeros(ComplexF64, length(controls), length(tlist)-1)
        for _ in objectives
    ]
    GrapeWrk(
        objectives,
        adjoint_objectives,
        kwargs,
        controls,
        pulsevals,
        gradient,
        searchdirection,
        result,
        bw_states,
        fw_states,
        fw_grad_states,
        tau_grads,
        G,
        TDgradG,
        gradG,
        control_derivs,
        vals_dict,
        bw_storage,
        fw_prop_wrk,
        bw_prop_wrk,
        grad_prop_wrk,
        use_threads
    )
end
