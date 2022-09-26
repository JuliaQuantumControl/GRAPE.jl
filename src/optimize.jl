using QuantumControlBase.QuantumPropagators:
    propstep!, write_to_storage!, get_from_storage!, set_state!, reinitprop!
using QuantumControlBase: resetgradvec!
using QuantumControlBase.Functionals: grad_J_T_via_chi!, make_gradient, make_chi
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

* `J_T`: A function `J_T(ϕ, objectives; τ=τ)` that evaluates the final time
  functional from a vector `ϕ` of forward-propagated states and
  `problem.objectives`. For all `objectives` that define a `target_state`, the
  element `τₖ` of the vector `τ` will contain the overlap of the state `ϕₖ`
  with the `target_state` of the `k`'th objective, or `NaN` otherwise.

# Optional problem keyword arguments

* `gradient_via`: A flag indicating how the gradient of `J_T` should be
   calculated. One of `:tau`, `:chi`. If all objectives in `problem` define a
   `target_function`, the default is `gradient_via=:tau`. This understands the
   functional `J_T` as a function of the complex overlaps between the
   propagated states and the target states, and evaluates the gradient via the
   chain rule. For functionals that cannot be expressed in terms of the overlap
   of propagated and target states, and/or if not all objectives define a
   target state, `gradient_via=:chi` because the default. In this case, the
   functional `J_T` is understood as a function of the forward-propagated
   states directly, and the full gradient is again calculated by the chain
   rule. See [`make_gradient`](@ref) for details.
* `gradient`:  A function to evaluate the gradient of `J_T`. By default, it is
  constructed via [`make_gradient`](@ref). If given manually, it must meet the
  interface described by [`make_gradient`](@ref). Most importantly, it must be
  consistent with the chosen `gradient_via`.
* `chi`: If `gradient_via=:chi`, a function that constructs the χ-states for
   the backward propagation, see [`make_gradient`](@ref) for details. By
   default, it is constructed via [`make_chi`](@ref). If given manually, it
   must meet the same interface described in [`make_chi`](@ref).
* `force_zygote=false`: Whether to force the use of automatic differentiation
  in [`make_gradient`](@ref) and [`make_chi`](@ref). This disables analytic
  gradients. The only reason to do this is for testing/benchmarking analytic vs
  automatic gradients.
* `update_hook`: Not implemented
* `info_hook`: A function that receives the same arguments as `update_hook`, in
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
* `verbose=false`: If `true`, print information during initialization

The propagation method for the forward propagation of each objective is
determined by the first available item of the following:

* a `fw_prop_method` keyword argument
* a `prop_method` keyword argument
* a property `fw_prop_method` of the objective
* a property `prop_method` of the objective
* the value `:auto`

The propagation method for the backward propagation is determined similarly,
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
    # TODO: always evaluate via chi
    info_hook = get(problem.kwargs, :info_hook, print_table)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)
    verbose = get(problem.kwargs, :verbose, false)

    wrk = GrapeWrk(problem; verbose)

    χ = wrk.chi_states
    Ψ₀ = [obj.initial_state for obj ∈ wrk.objectives]
    Ψtgt = Union{eltype(Ψ₀),Nothing}[
        (hasproperty(obj, :target_state) ? obj.target_state : nothing) for
        obj ∈ wrk.objectives
    ]

    if any(isnothing, Ψtgt)
        gradient_via = get(wrk.kwargs, :gradient_via, :chi)
        if gradient_via == :tau
            error("Evaluating gradients via τ requires target_states for all objectives")
        end
    else
        gradient_via = get(wrk.kwargs, :gradient_via, :tau)
    end
    allowed_gradient_via = (:tau, :chi)
    J_T_func = wrk.kwargs[:J_T]
    force_zygote = get(wrk.kwargs, :force_zygote, false)
    default_gradfunc! =
        make_gradient(J_T_func, wrk.objectives; via=gradient_via, force_zygote)
    gradfunc! = get(wrk.kwargs, :gradient, default_gradfunc!)
    if gradient_via == :tau
        # chi! is not used
    elseif gradient_via == :chi
        default_chi! = make_chi(J_T_func, wrk.objectives; force_zygote)
        chi! = get(wrk.kwargs, :chi, default_chi!)
        if gradfunc! ≢ grad_J_T_via_chi!
            @warn "gradient_via=:chi requires gradient=grad_J_T_via_chi! Ignoring passed gradient"
            gradfunc! = grad_J_T_via_chi!
        end
    else
        throw(
            ArgumentError(
                "gradient_via $(repr(gradient_via)) is not one of $(repr(allowed_gradient_via))"
            )
        )
    end

    τ = wrk.result.tau_vals
    ∇τ = wrk.tau_grads
    N_T = length(wrk.result.tlist) - 1
    N = length(wrk.objectives)
    L = length(wrk.controls)
    Φ = wrk.fw_storage

    # Calculate the functional only; optionally store.
    # Side-effects: set Ψ, τ, wrk.result.f_calls, wrk.fg_count
    function f(F, G, pulsevals; storage=nothing, count_call=true)
        @assert !isnothing(F)
        @assert isnothing(G)
        if count_call
            wrk.result.f_calls += 1
            wrk.fg_count[2] += 1
        end
        @threadsif wrk.use_threads for k = 1:N
            local Φₖ = isnothing(storage) ? nothing : storage[k]
            reinitprop!(wrk.fw_propagators[k], Ψ₀[k]; transform_control_ranges)
            (Φₖ !== nothing) && write_to_storage!(Φₖ, 1, Ψ₀[k])
            for n = 1:N_T  # `n` is the index for the time interval
                local Ψₖ = propstep!(wrk.fw_propagators[k])
                (Φₖ !== nothing) && write_to_storage!(Φₖ, n + 1, Ψₖ)
            end
            local Ψₖ = wrk.fw_propagators[k].state
            τ[k] = isnothing(Ψtgt[k]) ? NaN : (Ψtgt[k] ⋅ Ψₖ)
        end
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        return J_T_func(Ψ, wrk.objectives; τ=τ)
    end

    # Calculate the functional and the gradient G ≡ ∇J_T
    function fg!(F, G, pulsevals)

        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        wrk.result.fg_calls += 1
        wrk.fg_count[1] += 1

        # forward propagation and storage of states
        J_T_val = f(wrk.result.J_T, nothing, pulsevals; storage=Φ, count_call=false)

        # backward propagation of combined χ-state and gradient
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        if gradient_via == :tau
            for k = 1:N
                copyto!(χ[k], Ψtgt[k])
            end
        elseif gradient_via == :chi
            chi!(χ, Ψ, wrk.objectives)
        end
        @threadsif wrk.use_threads for k = 1:N
            local Ψₖ = wrk.fw_propagators[k].state
            local χ̃ₖ = wrk.bw_grad_propagators[k].state
            resetgradvec!(χ̃ₖ, χ[k])
            reinitprop!(wrk.bw_grad_propagators[k], χ̃ₖ; transform_control_ranges)
            for n = N_T:-1:1  # N_T is the number of time slices
                χ̃ₖ = propstep!(wrk.bw_grad_propagators[k])
                get_from_storage!(Ψₖ, Φ[k], n)
                for l = 1:L
                    ∇τ[k][l, n] = χ̃ₖ.grad_states[l] ⋅ Ψₖ
                end
                resetgradvec!(χ̃ₖ)
                set_state!(wrk.bw_grad_propagators[k], χ̃ₖ)
            end
        end

        gradfunc!(G, τ, ∇τ)
        return J_T_val

    end

    optimizer = get_optimizer(wrk)
    res = run_optimizer(optimizer, wrk, fg!, info_hook, check_convergence!)
    finalize_result!(wrk, res)
    return wrk.result

end


function transform_control_ranges(c, ϵ_min, ϵ_max, check)
    if check
        return (min(ϵ_min, 2 * ϵ_min), max(ϵ_max, 2 * ϵ_max))
    else
        return (min(ϵ_min, 5 * ϵ_min), max(ϵ_max, 5 * ϵ_max))
    end
end


function get_optimizer(wrk)
    n = length(wrk.pulsevals)
    m = 10 # TODO: kwarg for number of limited memory corrections
    optimizer = get(wrk.kwargs, :optimizer, LBFGSB.L_BFGS_B(n, m))
    return optimizer
end
