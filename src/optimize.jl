using QuantumControlBase.QuantumPropagators.Generators: Operator
using QuantumControlBase.QuantumPropagators.Controls: evaluate, evaluate!
using QuantumControlBase.QuantumPropagators: prop_step!, set_state!, reinit_prop!
using QuantumControlBase.QuantumPropagators.Storage: write_to_storage!, get_from_storage!
using QuantumGradientGenerators: resetgradvec!
using QuantumControlBase: make_chi, make_grad_J_a
using QuantumControlBase: @threadsif
using LinearAlgebra
using Printf

import QuantumControlBase: optimize

@doc raw"""
```julia
result = optimize(problem; method=:GRAPE, kwargs...)
```

optimizes the given
control [`problem`](@ref QuantumControlBase.ControlProblem) via the GRAPE
method, by minimizing the functional

```math
J(\{ϵ_{ln}\}) = J_T(\{|ϕ_k(T)⟩\}) + λ_a J_a(\{ϵ_{ln}\})
```

where the final time functional ``J_T`` depends explicitly on the
forward-propagated states and the running cost ``J_a`` depends explicitly on
pulse values ``ϵ_{nl}`` of the l'th control discretized on the n'th interval of
the time grid.

Returns a [`GrapeResult`](@ref).

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
  given, it will be automatically determined from `J_T` via [`make_chi`](@ref)
  with the default parameters.
* `J_a`: A function `J_a(pulsevals, tlist)` that evaluates running costs over
  the pulse values, where `pulsevals` are the vectorized values ``ϵ_{nl}``.
  If not given, the optimization will not include a running cost.
* `gradient_method=:gradgen`: One of `:gradgen` (default) or `:taylor`.
  With `gradient_method=:gradgen`, the gradient is calculated using
  [QuantumGradientGenerators]
  (https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl).
  With `gradient_method=:taylor`, it is evaluated via a Taylor series, see
  Eq. (20) in Kuprov and Rogers,  J. Chem. Phys. 131, 234108
  (2009) [KuprovJCP09](@cite).
* `taylor_grad_max_order=100`: If given with `gradient_method=:taylor`, the
  maximum number of terms in the Taylor series. If
  `taylor_grad_check_convergence=true` (default), if the Taylor series does not
  convergence within the given number of terms, throw an an error. With
  `taylor_grad_check_convergence=true`, this is the exact order of the Taylor
  series.
* `taylor_grad_tolerance=1e-16`: If given with `gradient_method=:taylor` and
  `taylor_grad_check_convergence=true`, stop the Taylor series when the norm of
  the term falls below the given tolerance. Ignored if
  `taylor_grad_check_convergence=false`.
* `taylor_grad_check_convergence=true`: If given as `true` (default), check the
  convergence after each term in the Taylor series an stop as soon as the norm
  of the term drops below the given number. If `false`, stop after exactly
  `taylor_grad_max_order` terms.
* `lambda_a=1`: A weight for the running cost `J_a`.
* `grad_J_a`: A function to calculate the gradient of `J_a`. If not given, it
  will be automatically determined.
* `upper_bound`: An upper bound for the value of any optimized control.
  Time-dependent upper bounds can be specified via `pulse_options`.
* `lower_bound`: A lower bound for the value of any optimized control.
  Time-dependent lower bounds can be specified via `pulse_options`.
* `pulse_options`: A dictionary that maps every control (as obtained by
  [`get_controls`](@ref
  QuantumControlBase.QuantumPropagators.Controls.get_controls) from the
  `problem.objectives`) to a dict with the following possible keys:

  - `:upper_bounds`: A vector of upper bound values, one for each intervals of
    the time grid. Values of `Inf` indicate an unconstrained upper bound for
    that time interval, respectively the global `upper_bound`, if given.
  - `:lower_bounds`: A vector of lower bound values. Values of `-Inf` indicate
    an unconstrained lower bound for that time interval,

* `update_hook`: Not implemented
* `info_hook`: A function (or tuple of functions) that receives the same
  arguments as `update_hook`, in order to write information about the current
  iteration to the screen or to a file. The default `info_hook` prints a table
  with convergence information to the screen. Runs after `update_hook`. The
  `info_hook` function may return a tuple, which is stored in the list of
  `records` inside the [`GrapeResult`](@ref) object.
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
for the backward propagation of the extended gradient vector for each objective
is determined from `grad_prop_method`, `fw_prop_method`, `prop_method` in order
of precedence.
"""
optimize(problem, method::Val{:GRAPE}) = optimize_grape(problem)
optimize(problem, method::Val{:grape}) = optimize_grape(problem)

"""
See [`optimize(problem; method=:GRAPE, kwargs...)`](@ref optimize(::Any, ::Val{:GRAPE})).
"""
function optimize_grape(problem)
    update_hook! = get(problem.kwargs, :update_hook, (args...) -> nothing)
    # TODO: implement update_hook
    # TODO: streamline the interface for info_hook
    # TODO: check if x_tol, f_tol, g_tol are used necessary / used correctly
    info_hook = get(problem.kwargs, :info_hook, print_table)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)
    verbose = get(problem.kwargs, :verbose, false)
    gradient_method = get(problem.kwargs, :gradient_method, :gradgen)
    taylor_grad_max_order = get(problem.kwargs, :taylor_grad_max_order, 100)
    taylor_grad_tolerance = get(problem.kwargs, :taylor_grad_tolerance, 1e-16)
    taylor_grad_check_convergence =
        get(problem.kwargs, :taylor_grad_check_convergence, true)

    wrk = GrapeWrk(problem; verbose)

    χ = wrk.chi_states
    Ψ₀ = [obj.initial_state for obj ∈ wrk.objectives]
    Ψtgt = Union{eltype(Ψ₀),Nothing}[
        (hasproperty(obj, :target_state) ? obj.target_state : nothing) for
        obj ∈ wrk.objectives
    ]

    J = wrk.J_parts
    tlist = wrk.result.tlist
    J_T_func = wrk.kwargs[:J_T]
    J_a_func = get(wrk.kwargs, :J_a, nothing)
    ∇J_T = wrk.grad_J_T
    ∇J_a = wrk.grad_J_a
    λₐ = get(wrk.kwargs, :lambda_a, 1.0)
    chi! = get(wrk.kwargs, :chi, make_chi(J_T_func, wrk.objectives))
    grad_J_a! = nothing
    if !isnothing(J_a_func)
        grad_J_a! = get(wrk.kwargs, :grad_J_a, make_grad_J_a(J_a_func, tlist))
    end

    τ = wrk.result.tau_vals
    ∇τ = wrk.tau_grads
    N_T = length(tlist) - 1
    N = length(wrk.objectives)
    L = length(wrk.controls)
    Φ = wrk.fw_storage

    # Calculate the functional only; optionally store.
    # Side-effects:
    # set Ψ, τ, wrk.result.f_calls, wrk.fg_count wrk.J_parts
    function f(F, G, pulsevals; storage=nothing, count_call=true)
        if pulsevals ≢ wrk.pulsevals
            # Ideally, the optimizer uses the original `pulsevals`. LBFGSB
            # does, but Optim.jl does not. Thus, for Optim.jl, we need to copy
            # back the values.
            wrk.pulsevals .= pulsevals
        end
        @assert !isnothing(F)
        @assert isnothing(G)
        if count_call
            wrk.result.f_calls += 1
            wrk.fg_count[2] += 1
        end
        @threadsif wrk.use_threads for k = 1:N
            local Φₖ = isnothing(storage) ? nothing : storage[k]
            reinit_prop!(wrk.fw_propagators[k], Ψ₀[k]; transform_control_ranges)
            (Φₖ !== nothing) && write_to_storage!(Φₖ, 1, Ψ₀[k])
            for n = 1:N_T  # `n` is the index for the time interval
                local Ψₖ = prop_step!(wrk.fw_propagators[k])
                (Φₖ !== nothing) && write_to_storage!(Φₖ, n + 1, Ψₖ)
            end
            local Ψₖ = wrk.fw_propagators[k].state
            τ[k] = isnothing(Ψtgt[k]) ? NaN : (Ψtgt[k] ⋅ Ψₖ)
        end
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        J[1] = J_T_func(Ψ, wrk.objectives; τ=τ)
        if !isnothing(J_a_func)
            J[2] = λₐ * J_a_func(pulsevals, tlist)
        end
        return sum(J)
    end

    # Calculate the functional and the gradient G ≡ ∇J_T
    # Side-effects:
    # as in f(...); wrk.grad_J_T, wrk.grad_J_a
    function fg_gradgen!(F, G, pulsevals)

        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        wrk.result.fg_calls += 1
        wrk.fg_count[1] += 1

        # forward propagation and storage of states
        J_val_guess = sum(wrk.J_parts)
        J_val = f(J_val_guess, nothing, pulsevals; storage=Φ, count_call=false)

        # backward propagation of combined χ-state and gradient
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        chi!(χ, Ψ, wrk.objectives; τ=τ)  # τ from f(...)
        @threadsif wrk.use_threads for k = 1:N
            local Ψₖ = wrk.fw_propagators[k].state
            local χ̃ₖ = wrk.bw_grad_propagators[k].state
            resetgradvec!(χ̃ₖ, χ[k])
            reinit_prop!(wrk.bw_grad_propagators[k], χ̃ₖ; transform_control_ranges)
            for n = N_T:-1:1  # N_T is the number of time slices
                χ̃ₖ = prop_step!(wrk.bw_grad_propagators[k])
                get_from_storage!(Ψₖ, Φ[k], n)
                for l = 1:L
                    ∇τ[k][l, n] = χ̃ₖ.grad_states[l] ⋅ Ψₖ
                end
                resetgradvec!(χ̃ₖ)
                set_state!(wrk.bw_grad_propagators[k], χ̃ₖ)
            end
        end

        _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
        copyto!(G, ∇J_T)
        if !isnothing(grad_J_a!)
            grad_J_a!(∇J_a, pulsevals, tlist)
            axpy!(λₐ, ∇J_a, G)
        end
        return J_val

    end

    # Calculate the functional and the gradient G ≡ ∇J_T
    # Side-effects:
    # as in f(...); wrk.grad_J_T, wrk.grad_J_a
    function fg_taylor!(F, G, pulsevals)

        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        wrk.result.fg_calls += 1
        wrk.fg_count[1] += 1

        # forward propagation and storage of states
        J_val_guess = sum(wrk.J_parts)
        J_val = f(J_val_guess, nothing, pulsevals; storage=Φ, count_call=false)

        # backward propagation of χ-state
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        chi!(χ, Ψ, wrk.objectives; τ=τ)  # τ from f(...)
        @threadsif wrk.use_threads for k = 1:N
            local Ψₖ = wrk.fw_propagators[k].state
            reinit_prop!(wrk.bw_propagators[k], χ[k]; transform_control_ranges)
            local χₖ = wrk.bw_propagators[k].state
            local Hₖ⁺ = wrk.adjoint_objectives[k].generator
            local Hₖₙ⁺ = wrk.taylor_genops[k]
            for n = N_T:-1:1  # N_T is the number of time slices
                # TODO: It would be cleaner to encapsulate this in a
                # propagator-like interface that can reuse the gradgen
                # structure instead of the taylor_genops, control_derivs, and
                # taylor_grad_states in wrk
                get_from_storage!(Ψₖ, Φ[k], n)
                for l = 1:L
                    local μₖₗ = wrk.control_derivs[k][l]
                    if isnothing(μₖₗ)
                        ∇τ[k][l, n] = 0.0
                    else
                        local ϵₙ⁽ⁱ⁾ = @view pulsevals[(n-1)*L+1:n*L]
                        local vals_dict = IdDict(
                            control => val for (control, val) ∈ zip(wrk.controls, ϵₙ⁽ⁱ⁾)
                        )
                        local μₗₖₙ = evaluate(μₖₗ, tlist, n; vals_dict)
                        evaluate!(Hₖₙ⁺, Hₖ⁺, tlist, n; vals_dict)
                        local χ̃ₗₖ = wrk.taylor_grad_states[l, k][1]
                        local ϕ_temp = wrk.taylor_grad_states[l, k][2:5]
                        local dt = tlist[n] - tlist[n+1]
                        @assert dt < 0.0
                        taylor_grad_step!(
                            χ̃ₗₖ,
                            χₖ,
                            Hₖₙ⁺,
                            μₗₖₙ,
                            dt,
                            ϕ_temp;
                            check_convergence=taylor_grad_check_convergence,
                            max_order=taylor_grad_max_order,
                            tolerance=taylor_grad_tolerance
                        )
                        ∇τ[k][l, n] = dot(χ̃ₗₖ, Ψₖ)
                    end
                end
                χₖ = prop_step!(wrk.bw_propagators[k])
            end
        end

        _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
        copyto!(G, ∇J_T)
        if !isnothing(grad_J_a!)
            grad_J_a!(∇J_a, pulsevals, tlist)
            axpy!(λₐ, ∇J_a, G)
        end
        return J_val

    end

    optimizer = get_optimizer(wrk)
    try
        if gradient_method == :gradgen
            run_optimizer(optimizer, wrk, fg_gradgen!, info_hook, check_convergence!)
        elseif gradient_method == :taylor
            run_optimizer(optimizer, wrk, fg_taylor!, info_hook, check_convergence!)
        else
            error("Invalid gradient_method=$(repr(gradient_method)) ∉ (:gradgen, :taylor)")
        end
    catch exc
        # Primarily, this is intended to catch Ctrl-C in interactive
        # optimizations
        exc_msg = sprint(showerror, exc)
        wrk.result.message = "Exception: $exc_msg"
    end

    finalize_result!(wrk)
    return wrk.result

end


function update_result!(wrk::GrapeWrk, i::Int64)
    res = wrk.result
    for (k, propagator) in enumerate(wrk.fw_propagators)
        copyto!(res.states[k], propagator.state)
    end
    res.J_T_prev = res.J_T
    res.J_T = wrk.J_parts[1]
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


"""Print optimization progress as a table.

This functions serves as the default `info_hook` for an optimization with
GRAPE.
"""
function print_table(wrk, iteration, args...)
    # TODO: make_print_table that precomputes headers and such, and maybe
    # allows for more options.
    # TODO: should we report ΔJ instead of ΔJ_T?

    J_T = wrk.result.J_T
    ΔJ_T = J_T - wrk.result.J_T_prev
    secs = wrk.result.secs

    headers = ["iter.", "J_T", "|∇J_T|", "ΔJ_T", "FG(F)", "secs"]
    if wrk.J_parts[2] ≠ 0.0
        headers = ["iter.", "J_T", "|∇J_T|", "|∇J_a|", "ΔJ_T", "FG(F)", "secs"]
    end

    iter_stop = "$(get(wrk.kwargs, :iter_stop, 5000))"
    width = Dict(
        "iter." => max(length("$iter_stop"), 6),
        "J_T" => 11,
        "|∇J_T|" => 11,
        "|∇J_a|" => 11,
        "|∇J|" => 11,
        "ΔJ" => 11,
        "ΔJ_T" => 11,
        "FG(F)" => 8,
        "secs" => 8,
    )

    if iteration == 0
        for header in headers
            w = width[header]
            print(lpad(header, w))
        end
        print("\n")
    end

    strs = [
        "$iteration",
        @sprintf("%.2e", J_T),
        @sprintf("%.2e", norm(wrk.grad_J_T)),
        (iteration > 0) ? @sprintf("%.2e", ΔJ_T) : "n/a",
        @sprintf("%d(%d)", wrk.fg_count[1], wrk.fg_count[2]),
        @sprintf("%.1f", secs),
    ]
    if wrk.J_parts[2] ≠ 0.0
        insert!(strs, 4, @sprintf("%.2e", norm(wrk.grad_J_a)))
    end
    for (str, header) in zip(strs, headers)
        w = width[header]
        print(lpad(str, w))
    end
    print("\n")
    flush(stdout)
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


# Evaluate `|Ψ̃̃ ≡ (∂ exp[-i Ĥ dt] / ∂ϵ) |Ψ⟩` with `μ̂ = ∂Ĥ/∂ϵ` via an expansion
# into a Taylor series. See Kuprov and Rogers,  J. Chem. Phys. 131, 234108
# (2009), Eq. (20). That equation can be rewritten in a recursive formula
#
# ```math
# |Ψ̃⟩ = \sum_{n=1}^{∞} \frac{(-i dt)^n}{n!} |Φₙ⟩
# ```
#
# with
#
# ```math
# \begin{align}
#   |Φ_1⟩ &= μ̂ |Ψ⟩                  \\
#   |ϕ_n⟩ &= μ̂ Ĥⁿ⁻¹ |Ψ⟩ + Ĥ |Φₙ₋₁⟩
# \end{align}
# ```
function taylor_grad_step!(
    Ψ̃,
    Ψ,
    Ĥ,
    μ̂,
    dt,           # positive for fw-prop, negative for bw-prop
    temp_states;  # need at least 4 states similar to Ψ
    check_convergence=true,
    max_order=100,
    tolerance=1e-16
)

    ϕₙ, ϕₙ₋₁, ĤⁿΨ, Ĥⁿ⁻¹Ψ = temp_states
    mul!(ϕₙ₋₁, μ̂, Ψ)
    mul!(Ĥⁿ⁻¹Ψ, Ĥ, Ψ)
    α = -1im * dt
    mul!(Ψ̃, α, ϕₙ₋₁)

    for n = 2:max_order

        mul!(ϕₙ, Ĥ, ϕₙ₋₁)               # matrix-vector product
        mul!(ϕₙ, μ̂, Ĥⁿ⁻¹Ψ, true, true)  # (added) matrix-vector product

        α *= -1im * dt / n
        mul!(Ψ̃, α, ϕₙ, true, true)      # (scaled) vector-vector sum
        if check_convergence
            r = abs(α * norm(ϕₙ))
            if r < tolerance
                return Ψ̃
            end
        end

        mul!(ĤⁿΨ, Ĥ, Ĥⁿ⁻¹Ψ)             # matrix-vector product
        ĤⁿΨ, Ĥⁿ⁻¹Ψ = Ĥⁿ⁻¹Ψ, ĤⁿΨ  # swap...
        ϕₙ, ϕₙ₋₁ = ϕₙ₋₁, ϕₙ      # .... without copy

    end

    if check_convergence && max_order > 1
        # should have returned inside the loop
        error("taylor_grad_step! did not converge within $max_order iterations")
    else
        return Ψ̃
    end

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
