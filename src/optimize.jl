using QuantumControlBase
using QuantumControlBase.ConditionalThreads: @threadsif
using QuantumPropagators
using LinearAlgebra
import Optim
using Dates
using Printf


"""Result object returned by [`optimize_grape`](@ref)."""
mutable struct GrapeResult{STST}
    tlist :: Vector{Float64}
    iter_start :: Int64  # the starting iteration number
    iter_stop :: Int64 # the maximum iteration number
    iter :: Int64  # the current iteration number
    secs :: Float64  # seconds that the last iteration took
    tau_vals :: Vector{ComplexF64}
    J_T :: Float64  # the current value of the final-time functional J_T
    J_T_prev :: Float64  # previous value of J_T
    guess_controls :: Vector{Vector{Float64}}
    optimized_controls :: Vector{Vector{Float64}}
    states :: Vector{STST}
    start_local_time :: DateTime
    end_local_time :: DateTime
    records :: Vector{Tuple}  # storage for info_hook to write data into at each iteration
    converged :: Bool
    f_calls :: Int64
    fg_calls :: Int64
    optim_res :: Union{Nothing, Optim.MultivariateOptimizationResults}
    message :: String

    function GrapeResult(problem)
        tlist = problem.tlist
        controls = getcontrols(problem.objectives)
        iter_start = get(problem.kwargs, :iter_start, 0)
        iter_stop = get(problem.kwargs, :iter_stop, 5000)
        iter = iter_start
        secs = 0
        tau_vals = zeros(ComplexF64, length(problem.objectives))
        guess_controls = [
            discretize(control, tlist) for control in controls
        ]
        J_T = 0.0
        J_T_prev = 0.0
        optimized_controls = [copy(guess) for guess in guess_controls]
        states = [similar(obj.initial_state) for obj in problem.objectives]
        start_local_time = now()
        end_local_time = now()
        records = Vector{Tuple}()
        converged = false
        message = "in progress"
        f_calls = 0
        fg_calls = 0
        optim_res = nothing
        new{eltype(states)}(
            tlist, iter_start, iter_stop, iter, secs, tau_vals, J_T, J_T_prev,
            guess_controls, optimized_controls, states, start_local_time,
            end_local_time, records, converged, f_calls, fg_calls, optim_res,
            message)
    end
end



Base.show(io::IO, r::GrapeResult) = print(io, "GrapeResult<$(r.message)>")
Base.show(io::IO, ::MIME"text/plain", r::GrapeResult) = print(io, """
GRAPE Optimization Result
-------------------------
- Started at $(r.start_local_time)
- Number of objectives: $(length(r.states))
- Number of iterations: $(max(r.iter - r.iter_start, 0))
- Number of pure func evals: $(r.f_calls)
- Number of func/grad evals: $(r.fg_calls)
- Value of functional: $(@sprintf("%.5e", r.J_T))
- Reason for termination: $(r.message)
- Ended at $(r.end_local_time) ($(r.end_local_time - r.start_local_time))""")


# GRAPE workspace (for internal use)
struct GrapeWrk{
        OT<:QuantumControlBase.AbstractControlObjective,
        AOT<:QuantumControlBase.AbstractControlObjective,
        KWT,
        CTRST<:Tuple,
        POT<:AbstractDict,
        STST,
        VDT,
        STORT,
        PRWT,
        PRGWT,
        GT,
        TDGT,
        GGT
    }

    # a copy of the objectives
    objectives :: Vector{OT}

    # the adjoint objectives, containing the adjoint generators for the
    # backward propagation
    adjoint_objectives :: Vector{AOT}

    # The kwargs from the control problem
    kwargs :: KWT

    # Tuple of the original controls (probably functions)
    controls :: CTRST

    optimizer :: Any # TODO

    pulsevals :: Vector{Float64}

    # Result object
    result :: GrapeResult{STST}

    #################################
    # scratch objects, per objective:

    # backward-propagated states
    # note: storage for fw-propagated states is in result.states
    bw_states :: Vector{STST}

    # forward-propagated states (functional evaluation only)
    fw_states :: Vector{STST}

    # foward-propagated grad-vectors
    fw_grad_states :: Vector{GradVector{STST}}

    # gradients ∂τₖ/ϵₗ(tₙ)
    tau_grads :: Vector{Matrix{ComplexF64}}

    # dynamical generator (normal propagation) at a particular point in time
    G :: Vector{GT}

    # dynamica generator for grad-propagation, time-dependent
    TDgradG :: Vector{TDGT}

    # dynamical generator for grad-propagation at a particular point in time
    gradG :: Vector{GGT}

    control_derivs :: Vector{Vector{Union{Function, Nothing}}}

    vals_dict :: Vector{VDT}

    bw_storage :: Vector{STORT}  # backward storage array (per objective)

    prop_wrk :: Vector{PRWT}  # for normal propagation

    prop_grad_wrk :: Vector{PRGWT}  # for gradient propagation

    use_threads :: Bool

    function GrapeWrk(problem::QuantumControlBase.ControlProblem)
        prop_method = get(problem.kwargs, :prop_method, Val(:auto))
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
        pulsevals = reshape(transpose(hcat(
            [discretize_on_midpoints(control, tlist)
            for control in controls]...
        )), :)
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
            pulsevals = reshape(transpose(hcat(
                [discretize_on_midpoints(control, wrk.result.tlist)
                for control in result.optimized_controls]...
            )), :)
        else
            result = GrapeResult(problem)
        end
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
        prop_wrk = [
            QuantumControlBase.initobjpropwrk(obj, tlist, prop_method;
                                              initial_state=obj.initial_state,
                                              kwargs...)
            for obj in objectives
        ]
        lbfgs_kwargs = Dict{Symbol, Any}()
        lbfgs_keys = (:memory_length, :alphaguess, :linesearch, :P, :precond,
                      :manifold, :scaleinvH0)
        for key in lbfgs_keys
            if key in keys(kwargs)
                if :optimizer in keys(kwargs)
                    @warn "keyword argument $(String(key)) will be ignored because due to custom optimizer"
                end
                val = kwargs[key]
                (key == :memory_length) && (key = :m)
                lbfgs_kwargs[key] = val
            end
        end
        optimizer = get(kwargs, :optimizer, Optim.LBFGS(;lbfgs_kwargs...))
        TDgradG = [TimeDependentGradGenerator(obj.generator) for obj in objectives]
        gradG = [evalcontrols(G̃_of_t, dummy_vals) for G̃_of_t ∈ TDgradG]
        prop_grad_wrk = [
            initpropwrk(Ψ̃, tlist, prop_method, gradG[k]; kwargs...)
            for (k, Ψ̃) in enumerate(fw_grad_states)
        ]
        tau_grads :: Vector{Matrix{ComplexF64}} = [
            zeros(ComplexF64, length(controls), length(tlist)-1)
            for _ in objectives
        ]
        new{
            eltype(objectives), # OT
            eltype(adjoint_objectives), # AOT
            typeof(kwargs), # KWT
            typeof(controls), # CTRST
            typeof(pulse_options), # POT
            typeof(objectives[1].initial_state), # STST
            eltype(vals_dict), # VDT
            eltype(bw_storage), # STORT
            eltype(prop_wrk), # PRWT
            eltype(prop_grad_wrk), # PRGWT
            eltype(G), # GT
            eltype(TDgradG), # TDGT
            eltype(gradG) # GGT
        }(
            objectives,
            adjoint_objectives,
            kwargs,
            controls,
            optimizer,
            pulsevals,
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
            prop_wrk,
            prop_grad_wrk,
            use_threads
        )
    end

end


"""Optimize the control problem using GRAPE.

```julia
result = optimize_grape(problem; kwargs...)
```

optimizes the given control problem, see
[`QuantumControlBase.ControlProblem`](@ref).

Keyword arguments that control the optimization are taken from the keyword
arguments used in the instantiation of `problem`. Any `kwargs` passed directly
to `optimize_pulses` will update (overwrite) the parameters in `problem`.

# Required problem keyword arguments

* `J_T`: A function `J_T(ϕ, objectives, τ=τ)` that evaluates the final time
  functional from a list `ϕ` of forward-propagated states and
  `problem.objectives`.
* `gradient`:  A function `gradient!(G, τ, ∇τ)` that stores the gradient of
  `J_T` in `G`.

# Optional problem keyword arguments

* `update_hook`
* `info_hook`
* `check_convergence`
* `x_tol`
* `f_tol`
* `g_tol`
* `show_trace`
* `extended_trace`
* `show_every`
* `allow_f_increases`
* `optimizer`

The following optional keyword arguments tune the default LBFGS optimizer. They
are ignored if a custom `optimizer` is passed.

* `memory_length`
* `alphaguess`
* `linesearch`
* `P`
* `precond`
* `manifold`
* `scaleinvH0`
"""
function optimize_grape(problem; kwargs...)
    merge!(problem.kwargs, kwargs)
    update_hook! = get(problem.kwargs, :update_hook, (args...) -> nothing)
    info_hook = get(problem.kwargs, :info_hook, print_table)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)

    wrk = GrapeWrk(problem)

    χ = wrk.bw_states
    Ψ = wrk.fw_states
    Ψ̃ = wrk.fw_grad_states
    τ = wrk.result.tau_vals
    ∇τ = wrk.tau_grads
    N_T = length(wrk.result.tlist) - 1
    N = length(wrk.objectives)
    L = length(wrk.controls)
    X = wrk.bw_storage

    gradfunc! = wrk.kwargs[:gradient]
    J_T_func = wrk.kwargs[:J_T]

    # calculate the functional only
    function f(F, G, pulsevals)
        @assert !isnothing(F)
        @assert isnothing(G)
        wrk.result.f_calls += 1
        @threadsif wrk.use_threads for k = 1:N
            copyto!(Ψ[k], wrk.objectives[k].initial_state)
            for n = 1:N_T  # `n` is the index for the time interval
                local (G, dt) = _fw_gen(pulsevals, k, n, wrk)
                propstep!(Ψ[k], G, dt, wrk.prop_wrk[k])
            end
        end
        return J_T_func(Ψ, wrk.objectives; τ=τ)
    end

    # calculate the functional and the gradient
    function fg!(F, G, pulsevals)
        wrk.result.fg_calls += 1
        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        # backward propagation of states
        @threadsif wrk.use_threads for k = 1:N
            copyto!(χ[k], wrk.objectives[k].target_state)
            write_to_storage!(X[k], N_T+1, χ[k])
            for n = N_T:-1:1
                local (G, dt) = _bw_gen(pulsevals, k, n, wrk)
                propstep!(χ[k], G, dt, wrk.prop_wrk[k])
                write_to_storage!(X[k], n, χ[k])
            end
        end
        # forward propagation of gradients
        @threadsif wrk.use_threads for k = 1:N
            resetgradvec!(Ψ̃[k], wrk.objectives[k].initial_state)
            for n = 1:N_T  # `n` is the index for the time interval
                local (G̃, dt) = _fw_gradgen(pulsevals, k, n, wrk)
                propstep!(Ψ̃[k], G̃, dt, wrk.prop_grad_wrk[k])
                get_from_storage!(χ[k], X[k], n+1)
                for l = 1:L
                    ∇τ[k][l, n] = dot(χ[k], Ψ̃[k].grad_states[l])
                end
                resetgradvec!(Ψ̃[k])
            end
            τ[k] = dot(wrk.objectives[k].target_state, Ψ̃[k].state)
        end
        gradfunc!(G, τ, ∇τ)
        # TODO: set wrk.result.states
        return J_T_func([Ψ̃[k].state for k in 1:N], wrk.objectives; τ=τ)
    end

    # update the result object and check convergence
    function callback(optim_state)
        iter = wrk.result.iter_start + optim_state.iteration
        update_result!(wrk, optim_state, iter)
        #update_hook!(...) # TODO
        info_tuple = info_hook(wrk, optim_state, wrk.result.iter)
        (info_tuple !== nothing) && push!(wrk.result.records, info_tuple)
        check_convergence!(wrk.result)
        return wrk.result.converged
    end

    res = Optim.optimize(
        Optim.only_fg!(fg!),
        wrk.pulsevals,
        wrk.optimizer,
        Optim.Options(
            callback=callback,
            iterations=wrk.result.iter_stop-wrk.result.iter_start, # TODO
            x_tol=get(wrk.kwargs, :x_tol, 0.0),
            f_tol=get(wrk.kwargs, :f_tol, 0.0),
            g_tol=get(wrk.kwargs, :g_tol, 1e-8),
            show_trace=get(wrk.kwargs, :show_trace, false),
            extended_trace=get(wrk.kwargs, :extended_trace, false),
            store_trace=get(wrk.kwargs, :store_trace, false),
            show_every=get(wrk.kwargs, :show_every, 1),
            allow_f_increases=get(wrk.kwargs, :allow_f_increases, false),
        )
    )

    finalize_result!(wrk, res)

    return wrk.result

end


# The dynamical generator for the forward propagation (functional evaluation)
function _fw_gen(pulse_vals, k, n, wrk)
    vals_dict = wrk.vals_dict[k]
    t = wrk.result.tlist
    L = length(wrk.controls)
    ϵ = reshape(pulse_vals, L, :)
    for (l, control) in enumerate(wrk.controls)
        vals_dict[control] = ϵ[l, n]
    end
    dt = t[n+1] - t[n]
    evalcontrols!(wrk.G[k], wrk.objectives[k].generator, vals_dict)
    return wrk.G[k], dt
end


# The dynamical generator for the gradient propagation
function _fw_gradgen(pulse_vals, k, n, wrk)
    vals_dict = wrk.vals_dict[k]
    t = wrk.result.tlist
    L = length(wrk.controls)
    ϵ = reshape(pulse_vals, L, :)
    for (l, control) in enumerate(wrk.controls)
        vals_dict[control] = ϵ[l, n]
    end
    dt = t[n+1] - t[n]
    evalcontrols!(wrk.gradG[k], wrk.TDgradG[k], vals_dict)
    return wrk.gradG[k], dt
end


# The dynamical generator for the backward propagation
function _bw_gen(pulse_vals, k, n, wrk)
    vals_dict = wrk.vals_dict[k]
    L = length(wrk.controls)
    ϵ = reshape(pulse_vals, L, :)
    t = wrk.result.tlist
    for (l, control) in enumerate(wrk.controls)
        vals_dict[control] = ϵ[l, n]
    end
    dt = t[n+1] - t[n]
    evalcontrols!(wrk.G[k], wrk.adjoint_objectives[k].generator, vals_dict)
    return wrk.G[k], -dt
end


function update_result!(wrk::GrapeWrk, optim_state, i::Int64)
    res = wrk.result
    res.J_T_prev = res.J_T
    res.J_T = optim_state.value
    (i > 0) && (res.iter = i)
    if i >= res.iter_stop
        res.converged = true
        res.message = "Reached maximum number of iterations"
        # Note: other convergence checks are done in user-supplied
        # check_convergence routine
    end
    prev_time = res.end_local_time
    res.end_local_time = now()
    res.secs = Dates.toms(res.end_local_time - prev_time) / 1000.0
end


function finalize_result!(wrk::GrapeWrk, optim_res::Optim.MultivariateOptimizationResults)
    L = length(wrk.controls)
    res = wrk.result
    if !optim_res.ls_success
        @error "optimization failed (linesearch)"
        res.message = "Failed linesearch"
    end
    if optim_res.stopped_by.f_increased
        @error "loss of monotonic convergence (try allow_f_increases=true)"
        res.message = "Loss of monotonic convergence"
    end
    if !res.converged
        @warn "Optimization failed to converge"
    end
    res.end_local_time = now()
    ϵ_opt = reshape(Optim.minimizer(optim_res), L, :)
    for l in 1:L
        res.optimized_controls[l] = discretize(ϵ_opt[l, :], res.tlist)
    end
    res.optim_res = optim_res
end


"""Print optimization progress as a table.

This functions serves as the default `info_hook` for an optimization with
GRAPE.
"""
function print_table(wrk, optim_state, iteration, args...)
    J_T = wrk.result.J_T
    ΔJ_T = J_T - wrk.result.J_T_prev
    secs = wrk.result.secs

    iter_stop = "$(get(wrk.kwargs, :iter_stop, 5000))"
    widths = [max(length("$iter_stop"), 6), 11, 11, 11, 8]

    if iteration == 0
        header = ["iter.", "J_T", "|∇J_T|", "ΔJ_T", "secs"]
        for (header, w) in zip(header, widths)
            print(lpad(header, w))
        end
        print("\n")
    end

    strs = (
        "$iteration",
        @sprintf("%.2e", J_T),
        @sprintf("%.2e", optim_state.g_norm),
        (iteration > 0) ? @sprintf("%.2e", ΔJ_T) : "n/a",
        @sprintf("%.1f", secs),
    )
    for (str, w) in zip(strs, widths)
        print(lpad(str, w))
    end
    print("\n")
end
