import QuantumControlBase
using QuantumControlBase.QuantumPropagators: init_storage, initprop
using QuantumControlBase.QuantumPropagators.Controls: getcontrols, discretize_on_midpoints
using QuantumControlBase: GradVector, TimeDependentGradGenerator
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
    pulsevals::Vector{Float64}

    gradient::Vector{Float64}  # gradient for guess in iterations

    searchdirection::Vector{Float64}  # search-direction for guess in iterations

    fg_count::Vector{Int64}

    # Resultt object
    result

    #################################
    # scratch objects, per objective:

    # backward-propagated states
    chi_states

    # gradients ∂τₖ/ϵₗ(tₙ)
    tau_grads::Vector{Matrix{ComplexF64}}

    # dynamical generator for grad-bw-propagation, time-dependent
    TDgradG  # TODO: rename gradgen

    fw_storage # backward storage array (per objective)

    fw_propagators # for normal forward propagation

    bw_grad_propagators  # for gradient backward propagation

    use_threads::Bool

end

function GrapeWrk(problem::QuantumControlBase.ControlProblem; verbose=false)
    use_threads = get(problem.kwargs, :use_threads, false)
    objectives = [obj for obj in problem.objectives]
    adjoint_objectives = [adjoint(obj) for obj in problem.objectives]
    controls = getcontrols(objectives)
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
    # TODO: store pulse_options in workspace, allow for things like bounds and
    # pulse parametrization
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
    searchdirection = zeros(length(pulsevals))
    dummy_vals = IdDict(control => 1.0 for (i, control) in enumerate(controls))
    fw_storage = [init_storage(obj.initial_state, tlist) for obj in objectives]
    kwargs[:piecewise] = true  # only accept piecewise propagators
    fw_prop_method = [
        Val(
            QuantumControlBase.get_objective_prop_method(
                obj,
                :fw_prop_method,
                :prop_method;
                kwargs...
            )
        ) for obj in objectives
    ]
    grad_prop_method = [
        Val(
            QuantumControlBase.get_objective_prop_method(
                obj,
                :grad_prop_method,
                :bw_prop_method,
                :prop_method;
                kwargs...
            )
        ) for obj in objectives
    ]

    fw_propagators = [
        begin
            verbose &&
                @info "Initializing fw-prop of objective $k with method $(fw_prop_method[k])"
            initprop(
                obj.initial_state,
                obj.generator,
                tlist;
                method=fw_prop_method[k],
                parameters=parameters,
                kwargs...
            )
        end for (k, obj) in enumerate(objectives)
    ]
    TDgradG = [TimeDependentGradGenerator(obj.generator) for obj in adjoint_objectives]
    chi_states = [similar(obj.initial_state) for obj in objectives]
    bw_grad_propagators = [
        begin
            verbose &&
                @info "Initializing gradient bw-prop of objective $k with method $(grad_prop_method[k])"
            χ̃ₖ = GradVector(chi_states[k], length(controls))
            initprop(
                χ̃ₖ,
                TDgradG[k],
                tlist;
                method=grad_prop_method[k],
                backward=true,
                parameters=parameters,
                kwargs...
            )
        end for k ∈ eachindex(objectives)
    ]
    tau_grads::Vector{Matrix{ComplexF64}} =
        [zeros(ComplexF64, length(controls), length(tlist) - 1) for _ in objectives]
    GrapeWrk(
        objectives,
        adjoint_objectives,
        kwargs,
        controls,
        pulsevals,
        gradient,
        searchdirection,
        fg_count,
        result,
        chi_states,
        tau_grads,
        TDgradG,
        fw_storage,
        fw_propagators,
        bw_grad_propagators,
        use_threads
    )
end
