using QuantumControlBase.QuantumPropagators:
    propstep!, write_to_storage!, get_from_storage!, set_state!, reinitprop!
using QuantumControlBase: resetgradvec!
using QuantumControlBase.Functionals: make_chi
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

* `chi`: A function `chi!(χ, ϕ, objectives)` what receives a list `ϕ`
  of the forward propagated states and must set ``|χₖ⟩ = -∂J_T/∂⟨ϕₖ|``. If not
  given, it will be automatically determined from `J_T` via [`make_chi`](@ref
  QuantumControlBase.Functionals.make_chi) with the default parameters.
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

    J_T_func = wrk.kwargs[:J_T]
    chi! = get(wrk.kwargs, :chi, make_chi(J_T_func, wrk.objectives))

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
        chi!(χ, Ψ, wrk.objectives; τ=τ)  # τ from f(...)
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

        _grad_J_T_via_chi!(G, τ, ∇τ)
        return J_T_val

    end

    optimizer = get_optimizer(wrk)
    res = run_optimizer(optimizer, wrk, fg!, info_hook, check_convergence!)
    finalize_result!(wrk, res)
    return wrk.result

end


# Gradient for an arbitrary functional evaluated via χ-states.
#
# ```julia
# _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
# ```
#
# sets the (vectorized) elements of the gradient `∇J_T` to the gradient
# ``∂J_T/∂ϵ_{ln}`` for an arbitrary functional ``J_T=J_T(\{|ϕ_k(T)⟩\})``, under
# the assumption that
#
# ```math
# \begin{aligned}
#     τ_k &= ⟨χ_k|ϕ_k(T)⟩ \quad \text{with} \quad |χ_k⟩ &= -∂J_T/∂⟨ϕ_k(T)|
#     \quad \text{and} \\
#     ∇τ_{kln} &= ∂τ_k/∂ϵ_{ln}\,,
# \end{aligned}
# ```
#
# where ``|ϕ_k(T)⟩`` is a state resulting from the forward propagation of some
# initial state ``|ϕ_k⟩`` under the pulse values ``ϵ_{ln}`` where ``l`` numbers
# the controls and ``n`` numbers the time slices. The ``τ_k`` are the elements
# of `τ` and ``∇τ_{kln}`` corresponds to `∇τ[k][l, n]`.
#
# In this case,
#
# ```math
# (∇J_T)_{ln} = ∂J_T/∂ϵ_{ln} = -2 \Re \sum_k ∇τ_{kln}\,.
# ```
#
# Note that the definition of the ``|χ_k⟩`` matches exactly the definition of
# the boundary condition for the backward propagation in Krotov's method, see
# [`QuantumControlBase.Functionals.make_chi`](@ref). Specifically, there is a
# minus sign in front of the derivative, compensated by the minus sign in the
# factor ``(-2)`` of the final ``(∇J_T)_{ln}``.
function _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
    N = length(τ) # number of objectives
    L, N_T = size(∇τ[1])  # number of controls/time intervals
    ∇J_T′ = reshape(∇J_T, L, N_T)  # writing to ∇J_T′ modifies ∇J_T
    for l = 1:L
        for n = 1:N_T
            ∇J_T′[l, n] = real(sum([∇τ[k][l, n] for k = 1:N]))
        end
    end
    lmul!(-2, ∇J_T)
    return ∇J_T
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
