using QuantumControlBase.QuantumPropagators: propstep!, write_to_storage!, get_from_storage!
using QuantumControlBase: evalcontrols!, resetgradvec!
using QuantumControlBase.ConditionalThreads: @threadsif
using LinearAlgebra
using Printf


"""Optimize a control problem using GRAPE.

```julia
result = optimize_grape(problem)
```

optimizes the given
control [`problem`](@ref QuantumControlBase.ControlProblem),
returning a [`GrapeResult`](@ref).

!!! note

    It is recommended to call [`optimize`](@ref QuantumControlBase.optimize)
    with `method=:GRAPE` instead of calling `optimize_grape` directly.

Keyword arguments that control the optimization are taken from the keyword
arguments used in the instantiation of `problem`.

# Required problem keyword arguments

* `J_T`: A function `J_T(ϕ, objectives, τ=τ)` that evaluates the final time
  functional from a list `ϕ` of forward-propagated states and
  `problem.objectives`.
* `gradient`:  A function `gradient!(G, τ, ∇τ)` that stores the gradient of
  `J_T` in `G`.

# Optional problem keyword arguments

* `update_hook`: Not immplemented
* `info_hook`: A function that receives the same argumens as `update_hook`, in
  order to write information about the current iteration to the screen or to a
  file. The default `info_hook` prints a table with convergence information to
  the screen. Runs after `update_hook`. The `info_hook` function may return a
  tuple, which is stored in the list of `records` inside the
  [`GrapeResult`](@ref) object.
* `check_convergence`: A function to check whether convergence has been
  reached. Receives a [`GrapeResult`](@ref) object `result`, and should set
  `result.converged` to `true` and `result.message` to an appropriate string in
  case of convergence. Multiple convergence checks can be performed by chaining
  functions with `∘`. The convergence check is performed after any calls to
  `update_hook` and `info_hook`.
* `x_tol`: Parameter for Optim.jl
* `f_tol`: Parameter for Optim.jl
* `g_tol`: Parameter for Optim.jl
* `show_trace`: Parameter for Optim.jl
* `extended_trace`:  Parameter for Optim.jl
* `show_every`: Parameter for Optim.jl
* `allow_f_increases`: Parameter for Optim.jl
* `optimizer`: An optional Optim.jl optimizer (`Optim.AbstractOptimizer`
  instance). If not given, an [L-BFGS-B](https://github.com/Gnimuc/LBFGSB.jl)
  optimizer will be used.
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
function optimize_grape(problem)
    update_hook! = get(problem.kwargs, :update_hook, (args...) -> nothing)
    # TODO: implement update_hook
    # TODO: streamline the interface for info_hook
    # TODO: check if x_tol, f_tol, g_tol are used necessary / used correctly
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
        wrk.fg_count[2] += 1
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
        wrk.fg_count[1] += 1
        # backward propagation of states
        @threadsif wrk.use_threads for k = 1:N
            copyto!(χ[k], wrk.objectives[k].target_state)
            write_to_storage!(X[k], N_T + 1, χ[k])
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
                get_from_storage!(χ[k], X[k], n + 1)
                for l = 1:L
                    ∇τ[k][l, n] = dot(χ[k], Ψ̃[k].grad_states[l])
                end
                resetgradvec!(Ψ̃[k])
            end
            τ[k] = dot(wrk.objectives[k].target_state, Ψ̃[k].state)
        end
        gradfunc!(G, τ, ∇τ)
        # TODO: set wrk.result.states
        return J_T_func([Ψ̃[k].state for k = 1:N], wrk.objectives; τ=τ)
    end

    optimizer = get_optimizer(wrk)
    res = run_optimizer(optimizer, wrk, fg!, info_hook, check_convergence!)
    finalize_result!(wrk, res)
    return wrk.result

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
