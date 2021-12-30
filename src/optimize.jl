using QuantumPropagators: propstep!, write_to_storage!, get_from_storage!
using QuantumControlBase: evalcontrols!
using QuantumControlBase: resetgradvec!
using QuantumControlBase.ConditionalThreads: @threadsif
using LinearAlgebra
using Printf


"""Optimize a control problem using GRAPE.

```julia
result = optimize_grape(problem; kwargs...)
```

optimizes the given
control [`problem`](@ref QuantumControlBase.ControlProblem),
returning a [`GrapeResult`](@ref).

!!! note

    It is recommended to call [`optimize`](@ref QuantumControlBase.optimize)
    with `method=:GRAPE` instead of calling `optimize_grape` directly.

Keyword arguments that control the optimization are taken from the keyword
arguments used in the instantiation of `problem`. Any `kwargs` passed directly
to `optimize_grape` will update (overwrite) the parameters in `problem`.

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
* `prop_method`/`fw_prop_method`/`bw_prop_method`: The propagation method to
  use for each objective, see below.
* `prop_method`/`fw_prop_method`/`grad_prop_method`: The propagation method to
  use for the extended gradient vector for each objective, see below.

The propagation method for the forward propagation of each objective is
determined by the first available item of the following:

* a `fw_prop_method` keyword argument
* a `prop_method` keyword argument
* a property `fw_prop_method` of the objective
* a property `prop_method` of the objective
* the value `:auto`

The propagation method for the backword propagation is determined similarly,
but with `bw_prop_method` instead of `fw_prop_method`. The propagation method
for the forward propagation of the extended gradient vector for each objective
is determined from `grad_prop_method`, `fw_prop_method`, `prop_method` in order
of precedence.
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
                propstep!(Ψ[k], G, dt, wrk.fw_prop_wrk[k])
            end
            τ[k] = dot(wrk.objectives[k].target_state, Ψ[k])
        end
        return J_T_func(Ψ, wrk.objectives; τ=τ)
    end

    # calculate the functional and the gradient
    function fg!(F, G, pulsevals)
        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        wrk.result.fg_calls += 1
        # backward propagation of states
        @threadsif wrk.use_threads for k = 1:N
            copyto!(χ[k], wrk.objectives[k].target_state)
            write_to_storage!(X[k], N_T+1, χ[k])
            for n = N_T:-1:1
                local (G, dt) = _bw_gen(pulsevals, k, n, wrk)
                propstep!(χ[k], G, dt, wrk.bw_prop_wrk[k])
                write_to_storage!(X[k], n, χ[k])
            end
        end
        # forward propagation of gradients
        @threadsif wrk.use_threads for k = 1:N
            resetgradvec!(Ψ̃[k], wrk.objectives[k].initial_state)
            for n = 1:N_T  # `n` is the index for the time interval
                local (G̃, dt) = _fw_gradgen(pulsevals, k, n, wrk)
                propstep!(Ψ̃[k], G̃, dt, wrk.grad_prop_wrk[k])
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

    optimizer = get_optimizer(wrk)
    res = run_optimizer(optimizer, wrk, fg!, info_hook, check_convergence!)
    finalize_result!(wrk, res)
    return wrk.result

end


function get_optimizer_optim_lbfgs(wrk) # TODO: get rid of this (not called)
    kwargs = wrk.kwargs
    lbfgs_kwargs = Dict{Symbol, Any}()
    lbfgs_keys = (:memory_length, :alphaguess, :linesearch, :P, :precond,
                    :manifold, :scaleinvH0)
    # TODO: get if of optim.jl-specific keywords: we'll default to LBFGS, and
    # if you want to use Optim.jl, you'll have to pass in a fully initialized
    # optimizer
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
    return optimizer
end


function get_optimizer(wrk)
    n = length(wrk.pulsevals)
    m = 10 # TODO: kwarg for number of limited memory corrections
    optimizer = get(wrk.kwargs, :optimizer, LBFGSB.L_BFGS_B(n, m))
    return optimizer
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
