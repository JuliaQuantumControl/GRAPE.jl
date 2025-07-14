# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using QuantumControl.QuantumPropagators.Controls: evaluate, evaluate!, discretize
using QuantumControl.QuantumPropagators: prop_step!, set_state!, reinit_prop!, propagate
using QuantumControl.QuantumPropagators.Storage:
    write_to_storage!, get_from_storage!, get_from_storage
using QuantumControl.QuantumPropagators.Interfaces: supports_inplace
using QuantumGradientGenerators: resetgradvec!
using QuantumControl: set_atexit_save_optimization, @threadsif
using QuantumControl.Functionals: make_chi, make_grad_J_a
using QuantumControl.QuantumPropagators: _StoreState
using LinearAlgebra
using Printf

import QuantumControl: optimize, make_print_iters

@doc raw"""
```julia
using GRAPE
result = optimize(problem; method=GRAPE, kwargs...)
```

optimizes the given control [`problem`](@ref QuantumControl.ControlProblem)
via the GRAPE method, by minimizing the functional

```math
J(\{ϵ_{nl}\}) = J_T(\{|Ψ_k(T)⟩\}) + λ_a J_a(\{ϵ_{nl}\})\,,
```

where the final time functional ``J_T`` depends explicitly on the
forward-propagated states ``|Ψ_k(T)⟩``, where ``|Ψ_k(t)⟩`` is the time
evolution of the `initial_state` in the ``k``th' trajectory in
`problem.trajectories`, and the running cost ``J_a`` depends explicitly on
pulse values ``ϵ_{nl}`` of the l'th control discretized on the n'th interval of
the time grid.

It does this by calculating the gradient of the final-time functional

```math
\nabla J_T \equiv \frac{\partial J_T}{\partial ϵ_{nl}}
= -2 \Re
\underbrace{%
\underbrace{\bigg\langle χ(T) \bigg\vert \hat{U}^{(k)}_{N_T} \dots \hat{U}^{(k)}_{n+1} \bigg \vert}_{\equiv \bra{\chi(t_n)}\;\text{(bw. prop.)}}
\frac{\partial \hat{U}^{(k)}_n}{\partial ϵ_{nl}}
}_{\equiv \bra{χ_k^\prime(t_{n-1})}}
\underbrace{\bigg \vert \hat{U}^{(k)}_{n-1} \dots \hat{U}^{(k)}_1 \bigg\vert Ψ_k(t=0) \bigg\rangle}_{\equiv |\Psi(t_{n-1})⟩\;\text{(fw. prop.)}}\,,
```

where ``\hat{U}^{(k)}_n`` is the time evolution operator for the ``n`` the
interval, generally assumed to be ``\hat{U}^{(k)}_n = \exp[-i \hat{H}_{kn}
dt_n]``, where ``\hat{H}_{kn}`` is the operator obtained by
evaluating `problem.trajectories[k].generator` on the ``n``'th time interval.

The backward-propagation of ``|\chi_k(t)⟩`` has the boundary condition

```math
    |\chi_k(T)⟩ \equiv - \frac{\partial J_T}{\partial ⟨\Psi_k(T)|}\,.
```

The final-time gradient ``\nabla J_T`` is combined with the gradient for the
running costs, and the total gradient is then fed into an optimizer
(L-BFGS-B by default) that iteratively changes the values ``\{ϵ_{nl}\}`` to
minimize ``J``.

See [Background](@ref GRAPE-Background) for details.

Returns a [`GrapeResult`](@ref).

Keyword arguments that control the optimization are taken from the keyword
arguments used in the instantiation of `problem`; any of these can be overridden
with explicit keyword arguments to `optimize`.


# Required problem keyword arguments

* `J_T`: A function `J_T(Ψ, trajectories)` that evaluates the final time
  functional from a list `Ψ` of forward-propagated states and
  `problem.trajectories`. The function `J_T` may also take a keyword argument
  `tau`. If it does, a vector containing the complex overlaps of the target
  states (`target_state` property of each trajectory in `problem.trajectories`)
  with the propagated states will be passed to `J_T`.

# Optional problem keyword arguments

* `chi`: A function `chi(Ψ, trajectories)` that receives a list `Ψ`
  of the forward propagated states and returns a vector of states
  ``|χₖ⟩ = -∂J_T/∂⟨Ψₖ|``. If not given, it will be automatically determined
  from `J_T` via [`QuantumControl.Functionals.make_chi`](@ref) with the
  default parameters. Similarly to `J_T`, if `chi` accepts a keyword argument
  `tau`, it will be passed a vector of complex overlaps.
* `chi_min_norm=1e-100`: The minimum allowable norm for any ``|χₖ(T)⟩``.
  Smaller norms would mean that the gradient is zero, and will abort the
  optimization with an error.
* `J_a`: A function `J_a(pulsevals, tlist)` that evaluates running costs over
  the pulse values, where `pulsevals` are the vectorized values ``ϵ_{nl}``,
  where `n` are in indices of the time intervals and `l` are the indices over
  the controls, i.e., `[ϵ₁₁, ϵ₂₁, …, ϵ₁₂, ϵ₂₂, …]` (the pulse values for each
  control are contiguous). If not given, the optimization will not include a
  running cost.
* `gradient_method=:gradgen`: One of `:gradgen` (default) or `:taylor`.
  With `gradient_method=:gradgen`, the gradient is calculated using
  [QuantumGradientGenerators](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl).
  With `gradient_method=:taylor`, it is evaluated via a Taylor series, see
  Eq. (20) in Kuprov and Rogers,  J. Chem. Phys. 131, 234108
  (2009) [KuprovJCP2009](@cite).
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
  will be automatically determined. See [`make_grad_J_a`](@ref) for the
  required interface.
* `upper_bound`: An upper bound for the value of any optimized control.
  Time-dependent upper bounds can be specified via `pulse_options`.
* `lower_bound`: A lower bound for the value of any optimized control.
  Time-dependent lower bounds can be specified via `pulse_options`.
* `pulse_options`: A dictionary that maps every control (as obtained by
  [`get_controls`](@ref
  QuantumControl.QuantumPropagators.Controls.get_controls) from the
  `problem.trajectories`) to a dict with the following possible keys:

  - `:upper_bounds`: A vector of upper bound values, one for each intervals of
    the time grid. Values of `Inf` indicate an unconstrained upper bound for
    that time interval, respectively the global `upper_bound`, if given.
  - `:lower_bounds`: A vector of lower bound values. Values of `-Inf` indicate
    an unconstrained lower bound for that time interval,

* `print_iters=true`: Whether to print information after each iteration.
* `print_iter_info=["iter.", "J_T", "|∇J|", "|Δϵ|", "ΔJ", "FG(F)", "secs"]`:
  Which fields to print if `print_iters=true`. If given, must be a list of
  header labels (strings), which can be any of the following:

  - `"iter."`: The iteration number
  - `"J_T"`: The value of the final-time functional for the dynamics under the
    optimized pulses
  - `"J_a"`: The value of the pulse-dependent running cost for the optimized
    pulses
  - `"λ_a⋅J_a"`: The total contribution of `J_a` to the full functional `J`
  - `"J"`: The value of the optimization functional for the optimized pulses
  - `"ǁ∇J_Tǁ"`: The ℓ²-norm of the *current* gradient of the final-time
    functional. Note that this is usually the gradient of the optimize pulse,
    not the guess pulse.
  - `"ǁ∇J_aǁ"`: The ℓ²-norm of the the *current* gradient of the pulse-dependent
    running cost. For comparison with `"ǁ∇J_Tǁ"`.
  - `"λ_aǁ∇J_aǁ"`: The ℓ²-norm of the the *current* gradient of the complete
    pulse-dependent running cost term. For comparison with `"ǁ∇J_Tǁ"`.
  - `"ǁ∇Jǁ"`: The norm of the guess pulse gradient. Note that the *guess* pulse
    gradient is not the same the *current* gradient.
  - `"ǁΔϵǁ"`:  The ℓ²-norm of the pulse update
  - `"ǁϵǁ"`: The ℓ²-norm of optimized pulse values
  - `"max|Δϵ|"` The maximum value of the pulse update (infinity norm)
  - `"max|ϵ|"`: The maximum value of the pulse values (infinity norm)
  - `"ǁΔϵǁ/ǁϵǁ"`: The ratio of the pulse update tothe optimized pulse values
  - `"∫Δϵ²dt"`: The L²-norm of the pulse update, summed over all pulses. A
    convergence measure comparable (proportional) to the running cost in
    Krotov's method
  - `"ǁsǁ"`: The norm of the search direction. Should be `ǁΔϵǁ` scaled by the
    step with `α`.
  - `"∠°"`: The angle (in degrees) between the negative gradient `-∇J` and the
    search direction `s`.
  - `"α"`: The step width as determined by the line search (`Δϵ = α⋅s`)
  - `"ΔJ_T"`: The change in the final time functional relative to the previous
    iteration
  - `"ΔJ_a"`:  The change in the control-dependent running cost relative to the
    previous iteration
  - `"λ_a⋅ΔJ_a"`: The change in the control-dependent running cost term
    relative to the previous iteration.
  - `"ΔJ"`:  The change in the total optimization functional relative to the
    previous iteration.
  - `"FG(F)"`:  The number of functional/gradient evaluation (FG), or pure
    functional (F) evaluations
  - `"secs"`:  The number of seconds of wallclock time spent on the iteration.

  * `store_iter_info=[]`: Which fields to store in `result.records`, given as
  a list of header labels, see `print_iter_info`.
* `callback`: A function (or tuple of functions) that receives the
  [GRAPE workspace](@ref GrapeWrk) and the iteration number. The function
  may return a tuple of values which are stored in the
  [`GrapeResult`](@ref) object `result.records`. The function can also mutate
  the workspace, in particular the updated `pulsevals`. This may be used,
  e.g., to apply a spectral filter to the updated pulses or to perform
  similar manipulations. Note that `print_iters=true` (default) adds an
  automatic callback to print information after each iteration. With
  `store_iter_info`, that callback automatically stores a subset of the
  available information.
* `check_convergence`: A function to check whether convergence has been
  reached. Receives a [`GrapeResult`](@ref) object `result`, and should set
  `result.converged` to `true` and `result.message` to an appropriate string in
  case of convergence. Multiple convergence checks can be performed by chaining
  functions with `∘`. The convergence check is performed after any `callback`.
* `prop_method`: The propagation method to use for each trajectory, see below.
* `verbose=false`: If `true`, print information during initialization
* `rethrow_exceptions`: By default, any exception ends the optimization, but
  still returns a [`GrapeResult`](@ref) that captures the message associated
  with the exception. This is to avoid losing results from a long-running
  optimization when an exception occurs in a later iteration. If
  `rethrow_exceptions=true`, instead of capturing the exception, it will be
  thrown normally.

# Experimental keyword arguments

The following keyword arguments may change in non-breaking releases:

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

# Trajectory propagation

GRAPE may involve three types of propagation:

* A forward propagation for every [`Trajectory`](@ref) in the `problem`
* A backward propagation for every trajectory
* A backward propagation of a
  [gradient generator](@extref QuantumGradientGenerators.GradGenerator)
  for every trajectory.

The keyword arguments for each propagation (see [`propagate`](@ref)) are
determined from any properties of each [`Trajectory`](@ref) that have a `prop_`
prefix, cf. [`init_prop_trajectory`](@ref).

In situations where different parameters are required for the forward and
backward propagation, instead of the `prop_` prefix, the `fw_prop_` and
`bw_prop_` prefix can be used, respectively. These override any setting with
the `prop_` prefix. Similarly, properties for the backward propagation of the
gradient generators can be set with properties that have a `grad_prop_` prefix.
These prefixes apply both to the properties of each [`Trajectory`](@ref) and
the problem keyword arguments.

Note that the propagation method for each propagation must be specified. In
most cases, it is sufficient (and recommended) to pass a global `prop_method`
problem keyword argument.
"""
optimize(problem, method::Val{:GRAPE}) = optimize_grape(problem)
optimize(problem, method::Val{:grape}) = optimize_grape(problem)

"""
See [`optimize(problem; method=GRAPE, kwargs...)`](@ref optimize(::Any, ::Val{:GRAPE})).
"""
function optimize_grape(problem)
    # TODO: check if x_tol, f_tol, g_tol are used necessary / used correctly
    callback = get(problem.kwargs, :callback, (args...) -> nothing)
    if haskey(problem.kwargs, :update_hook) || haskey(problem.kwargs, :info_hook)
        msg = "The `update_hook` and `info_hook` arguments have been superseded by the `callback` argument"
        throw(ArgumentError(msg))
    end
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)
    verbose = get(problem.kwargs, :verbose, false)

    wrk = GrapeWrk(problem; verbose)

    tlist = wrk.result.tlist
    J_a_func = get(wrk.kwargs, :J_a, nothing)
    if isnothing(J_a_func)
        if haskey(wrk.kwargs, :grad_J_a)
            @warn "Argument `grad_J_A` was given without `grad_J_a`. Ignoring"
            delete!(wrk.kwargs, :grad_J_a)
        end
    else
        if !haskey(wrk.kwargs, :grad_J_a)
            wrk.kwargs[:grad_J_a] = make_grad_J_a(J_a_func, tlist)
        end
    end

    function f(F, G, pulsevals)
        # Closure around `problem` (read-only) and `wrk` (read-write)`
        @assert !isnothing(F)
        @assert isnothing(G)
        return evaluate_functional(pulsevals, problem, wrk)
    end

    function fg!(F, G, pulsevals)
        # Closure around `problem` (read-only) and `wrk` (read-write)`
        if isnothing(G)  # functional only
            return evaluate_functional(pulsevals, problem, wrk)
        end
        return evaluate_gradient!(G, pulsevals, problem, wrk)
    end

    optimizer = wrk.optimizer
    atexit_filename = get(problem.kwargs, :atexit_filename, nothing)
    # atexit_filename is undocumented on purpose: this is considered a feature
    # of @optimize_or_load
    if !isnothing(atexit_filename)
        set_atexit_save_optimization(atexit_filename, wrk.result)
        if !isinteractive()
            @info "Set callback to store result in $(relpath(atexit_filename)) on unexpected exit."
            # In interactive mode, `atexit` is very unlikely, and
            # `InterruptException` is handles via try/catch instead.
        end
    end
    try
        run_optimizer(optimizer, wrk, fg!, callback, check_convergence!)
    catch exc
        if get(problem.kwargs, :rethrow_exceptions, false)
            rethrow()
        end
        # Primarily, this is intended to catch Ctrl-C in interactive
        # optimizations (InterruptException)
        exc_msg = sprint(showerror, exc)
        wrk.result.message = "Exception: $exc_msg"
    end

    finalize_result!(wrk)
    if !isnothing(atexit_filename)
        popfirst!(Base.atexit_hooks)
    end

    return wrk.result

end


function run_optimizer(optimizer, args...)
    error("Unknown optimizer: $optimizer")
    # The methods for different optimizers are implemented as module extensions
    # for LBFGS and Optim.
end


function update_result!(wrk::GrapeWrk, i::Int64)
    res = wrk.result
    for (k, propagator) in enumerate(wrk.fw_propagators)
        copyto!(res.states[k], propagator.state)
    end
    res.J_T_prev = res.J_T
    res.J_T = wrk.J_parts[1]
    res.J_a_prev = res.J_a
    res.J_a = wrk.J_parts[2]
    if res.J_a > 0.0
        λₐ = get(wrk.kwargs, :lambda_a, 1.0)
        res.J_a /= λₐ
    end
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


function finalize_result!(wrk::GrapeWrk)
    L = length(wrk.controls)
    res = wrk.result
    res.end_local_time = now()
    N_T = length(res.tlist) - 1
    for l = 1:L
        ϵ_opt = wrk.pulsevals[((l-1)*N_T+1):(l*N_T)]
        res.optimized_controls[l] = discretize(ϵ_opt, res.tlist)
    end
end


make_print_iters(::Val{:GRAPE}; kwargs...) = make_grape_print_iters(; kwargs...)
make_print_iters(::Val{:grape}; kwargs...) = make_grape_print_iters(; kwargs...)


"""Print optimization progress as a table.

This functions serves as the default `info_hook` for an optimization with
GRAPE.
"""
function make_grape_print_iters(; kwargs...)
    headers = [
        "iter.",
        "J_T",
        "J_a",
        # "J_b",
        "λ_a⋅J_a",
        # "λ_b⋅J_b",
        "J",
        "ǁ∇J_Tǁ",
        "ǁ∇J_aǁ",
        "λ_aǁ∇J_aǁ",
        # "ǁ∇J_bǁ",
        "λ_a⋅ΔJ_a",
        # "λ_b⋅ΔJ_b",
        "ǁ∇Jǁ",
        "ǁΔϵǁ",
        "ǁϵǁ",
        "max|Δϵ|",
        "max|ϵ|",
        "ǁΔϵǁ/ǁϵǁ",
        "∫Δϵ²dt",
        "ǁsǁ",
        "∠°",
        "α",
        "ΔJ_T",
        "ΔJ_a",
        # "ΔJ_b",
        "λ_a⋅ΔJ_a",
        # "λ_b⋅ΔJ_b",
        "ΔJ",
        "FG(F)",
        "secs"
    ]  # TODO: update docs when J_b is implemented
    delta_headers = Set([
        "ΔJ_T",
        "λ_a⋅ΔJ_a",
        "ΔJ_a",
        "λ_b⋅ΔJ_b",
        "ΔJ",
        "ǁΔϵǁ",
        "ǁΔϵǁ/ǁϵǁ",
        "max|Δϵ|",
        "∫Δϵ²dt",
        "α",
        "ǁsǁ"
    ])
    store_iter_info = get(kwargs, :store_iter_info, String[])
    if Set(store_iter_info) ⊈ Set(headers)
        diff = [field for field in store_iter_info if field ∉ headers]
        msg = "store_iter_info contains invalid elements $(diff)"
        @warn "Invalid $(diff) not in allowed fields = [$(join(map(repr, headers), ", "))]"
        throw(ArgumentError(msg))
    end
    print_iter_info = get(
        kwargs,
        :print_iter_info,
        ["iter.", "J_T", "ǁ∇Jǁ", "ǁΔϵǁ", "ΔJ", "FG(F)", "secs"]
    )
    if Set(print_iter_info) ⊈ Set(headers)
        diff = [field for field in print_iter_info if field ∉ headers]
        msg = "print_iter_info contains invalid elements $(diff)"
        @warn "Invalid $(diff) not in allowed fields = [$(join(map(repr, headers), ", "))]"
        throw(ArgumentError(msg))
    end
    needed_fields = Set(store_iter_info) ∪ Set(print_iter_info)
    info_vals = Dict{String,Any}()

    function print_table(wrk, iteration, args...)

        λ_a = get(wrk.kwargs, :lambda_a, 1.0)
        info_vals["iter."] = iteration
        info_vals["J_T"] = wrk.result.J_T
        info_vals["ΔJ_T"] = wrk.result.J_T - wrk.result.J_T_prev
        info_vals["J_a"] = wrk.result.J_a
        info_vals["λ_a⋅J_a"] = wrk.J_parts[2]
        ΔJ_a = wrk.result.J_a - wrk.result.J_a_prev
        info_vals["ΔJ_a"] = ΔJ_a
        info_vals["λ_a⋅ΔJ_a"] = λ_a * ΔJ_a
        info_vals["J"] = wrk.result.J_T + λ_a * wrk.result.J_a
        if "ǁ∇J_Tǁ" ∈ needed_fields
            info_vals["ǁ∇J_Tǁ"] = norm(wrk.grad_J_T)
        end
        if ("ǁ∇J_aǁ" ∈ needed_fields) || ("λ_aǁ∇J_aǁ" ∈ needed_fields)
            nrm_∇J_a = norm(wrk.grad_J_a)
            info_vals["ǁ∇J_aǁ"] = nrm_∇J_a
            info_vals["λ_aǁ∇J_aǁ"] = λ_a * nrm_∇J_a
        end
        if "ǁ∇Jǁ" ∈ needed_fields
            info_vals["ǁ∇Jǁ"] = norm(gradient(wrk; which=:initial))
        end
        if "ΔJ" ∈ needed_fields
            J = wrk.result.J_T + λ_a * wrk.result.J_a
            J_prev = wrk.result.J_T_prev + λ_a * wrk.result.J_a_prev
            info_vals["ΔJ"] = J - J_prev
        end
        if ("ǁΔϵǁ/ǁϵǁ" ∈ needed_fields) ||
           ("ǁΔϵǁ" ∈ needed_fields) ||
           ("ǁϵǁ" ∈ needed_fields) ||
           ("max|ϵ|" ∈ needed_fields) ||
           ("max|Δϵ|" ∈ needed_fields) ||
           ("∫Δϵ²dt" ∈ needed_fields)
            r = 0.0
            rΔ = 0.0
            ∫Δϵ²dt = 0.0
            max_ϵ = 0.0
            max_Δϵ = 0.0
            N = length(wrk.result.tlist) - 1
            for i = 1:length(wrk.pulsevals)
                n = ((i - 1) % N) + 1  # index of time interval 1…N
                dt = wrk.result.tlist[n+1] - wrk.result.tlist[n]
                r += wrk.pulsevals[i]^2
                abs_Δϵᵢ = abs(wrk.pulsevals[i] - wrk.pulsevals_guess[i])
                Δϵᵢ² = abs_Δϵᵢ^2
                ∫Δϵ²dt += Δϵᵢ² * dt
                rΔ += Δϵᵢ²
                abs_ϵᵢ = abs(wrk.pulsevals[i])
                (abs_ϵᵢ > max_ϵ) && (max_ϵ = abs_ϵᵢ)
                (abs_Δϵᵢ > max_Δϵ) && (max_Δϵ = abs_Δϵᵢ)
            end
            info_vals["ǁϵǁ"] = sqrt(r)
            info_vals["ǁΔϵǁ"] = sqrt(rΔ)
            info_vals["ǁΔϵǁ/ǁϵǁ"] = sqrt(rΔ) / sqrt(r)
            info_vals["max|ϵ|"] = max_ϵ
            info_vals["max|Δϵ|"] = max_Δϵ
            info_vals["∫Δϵ²dt"] = ∫Δϵ²dt
        end
        if "ǁsǁ" ∈ needed_fields
            info_vals["ǁsǁ"] = norm_search(wrk)
        end
        if "α" ∈ needed_fields
            info_vals["α"] = step_width(wrk)
        end
        if "∠°" ∈ needed_fields
            s_G = -1 * gradient(wrk; which=:initial)
            s = search_direction(wrk)
            info_vals["∠°"] = vec_angle(s_G, s; unit=:degree)
        end
        info_vals["FG(F)"] = Tuple(wrk.fg_count)
        info_vals["secs"] = wrk.result.secs

        iter_stop = "$(get(wrk.kwargs, :iter_stop, 5000))"
        width = Dict(  # default width is 11
            "iter." => max(length("$iter_stop"), 6),
            "FG(F)" => 8,
            "secs" => 8,
            "∠°" => 7,
        )

        if length(print_iter_info) > 0

            if iteration == 0
                for header in print_iter_info
                    w = get(width, header, 11)
                    print(lpad(header, w))
                end
                print("\n")
            end

            for header in print_iter_info
                if header == "iter."
                    str = "$(info_vals[header])"
                elseif header == "FG(F)"
                    counts = info_vals[header]
                    str = @sprintf("%d(%d)", counts[1], counts[2])
                elseif header == "secs"
                    str = @sprintf("%.1f", info_vals[header])
                elseif header ∈ delta_headers
                    if iteration > 0
                        str = @sprintf("%.2e", info_vals[header])
                    else
                        str = "n/a"
                    end
                elseif header == "∠°"
                    if iteration > 0
                        str = @sprintf("%.1f", info_vals["∠°"])
                    else
                        str = "n/a"
                    end
                else
                    str = @sprintf("%.2e", info_vals[header])
                end
                w = get(width, header, 11)
                print(lpad(str, w))
            end

            print("\n")
            flush(stdout)

        end

        return Tuple((info_vals[field] for field in store_iter_info))

    end

    return print_table

end


# Gradient for an arbitrary functional evaluated via χ-states.
#
# ```julia
# _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
# ```
#
# sets the (vectorized) elements of the gradient `∇J_T` to the gradient
# ``∂J_T/∂ϵ_{nl}`` for an arbitrary functional ``J_T=J_T(\{|ϕ_k(T)⟩\})``, under
# the assumption that
#
# ```math
# \begin{aligned}
#     τ_k &= ⟨χ_k|ϕ_k(T)⟩ \quad \text{with} \quad |χ_k⟩ &= -∂J_T/∂⟨ϕ_k(T)|
#     \quad \text{and} \\
#     ∇τ_{knl} &= ∂τ_k/∂ϵ_{nl}\,,
# \end{aligned}
# ```
#
# where ``|ϕ_k(T)⟩`` is a state resulting from the forward propagation of some
# initial state ``|ϕ_k⟩`` under the pulse values ``ϵ_{nl}`` where ``l`` numbers
# the controls and ``n`` numbers the time slices. The ``τ_k`` are the elements
# of `τ` and ``∇τ_{knl}`` corresponds to `∇τ[k][n, l]`.
#
# In this case,
#
# ```math
# (∇J_T)_{nl} = ∂J_T/∂ϵ_{nl} = -2 \Re \sum_k ∇τ_{knl}\,.
# ```
#
# Note that the definition of the ``|χ_k⟩`` matches exactly the definition of
# the boundary condition for the backward propagation in Krotov's method, see
# [`QuantumControl.Functionals.make_chi`](@ref). Specifically, there is a
# minus sign in front of the derivative, compensated by the minus sign in the
# factor ``(-2)`` of the final ``(∇J_T)_{nl}``.
function _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
    N = length(τ) # number of trajectories
    N_T, L = size(∇τ[1])  # number of time intervals/controls
    for l = 1:L
        for n = 1:N_T
            ∇J_T[(l-1)*N_T+n] = real(sum([∇τ[k][n, l] for k = 1:N]))
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
# TODO: this should probably be adapted to static states (avoiding in-place)
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


"""
Evaluate the optimization functional in `problem` for the given `pulsevals`.

```julia
J = evaluate_functional(pulsevals, problem, wrk; storage=nothing, count_call=true)
```

evaluates the functional defined in `problem`, for the given pulse values,
using `wrk.fw_propagators`, where `wrk` is the GRAPE workspace initialized from
`problem`. The `pulsevals` is a vector of `Float64` values corresponding to a
concatenation of all the controls in `problem`, discretized to the midpoints of
the time grid, cf. [`GrapeWrk`](@ref).

As a side effect, the evaluation sets the following information in `wrk`:

* `wrk.pulsevals`: On output, the values of the given `pulsevals`. Note that
  `pulsevals` may alias `wrk.pulsevals`, so there is no assumption made on
  `wrk.pulsevals` other than that mutating `wrk.pulsevals` directly affects the
  propagators in `wrk`.
* `wrk.result.f_calls`: Will be incremented by one (only if `count_call=true`)
* `wrk.fg_count[2]`: Will be incremented by one (only if `count_call=true`)
* `wrk.result.tau_vals`: For any trajectory that defines a `target_state`, the
  overlap of the propagated state with that target state.
* `wrk.J_parts`: The parts (`J_T`, `λₐJ_a`) of the functional

If `storage` is given, as a vector of storage containers suitable for
[`propagate`](@ref) (one for each trajectory), the forward-propagated states
    will be stored there.

Returns `J` as `sum(wrk.J_parts)`.
"""
function evaluate_functional(pulsevals, problem, wrk; storage=nothing, count_call=true)
    J_T = problem.kwargs[:J_T]
    J_a = get(problem.kwargs, :J_a, nothing)
    λₐ = get(problem.kwargs, :lambda_a, 1.0)
    trajectories = problem.trajectories
    N = length(trajectories)
    tlist = problem.tlist
    N_T = length(tlist) - 1  # number of time steps
    if pulsevals ≢ wrk.pulsevals
        # Ideally, the optimizer uses the original `pulsevals`. LBFGSB
        # does, but Optim.jl does not. Thus, for Optim.jl, we need to copy
        # back the values. In any case, setting `wrk.pulsevals` is how we
        # inject the pulse values into the propagation: all the propagators in
        # `wrk` are set up to alias `wrk.pulsevals`.
        wrk.pulsevals .= pulsevals
    end
    if count_call
        wrk.result.f_calls += 1
        wrk.fg_count[2] += 1
    end
    Ψ₀(k) = trajectories[k].initial_state
    Ψtgt(k) = trajectories[k].target_state
    @threadsif wrk.use_threads for k = 1:N
        local Φₖ = isnothing(storage) ? nothing : storage[k]
        reinit_prop!(wrk.fw_propagators[k], Ψ₀(k); transform_control_ranges)
        (Φₖ !== nothing) && write_to_storage!(Φₖ, 1, Ψ₀(k))
        # The optional storage exists so that `evaluate_functional` can be used
        # as part of `evaluate_gradient!`.
        for n = 1:N_T  # `n` is the index for the time interval
            local Ψₖ = prop_step!(wrk.fw_propagators[k])
            if haskey(wrk.fw_prop_kwargs[k], :callback)
                local cb = wrk.fw_prop_kwargs[k][:callback]
                local observables = get(wrk.fw_prop_kwargs[k], :observables, _StoreState())
                cb(wrk.fw_propagators[k], observables)
            end
            (Φₖ !== nothing) && write_to_storage!(Φₖ, n + 1, Ψₖ)
        end
        local Ψₖ = wrk.fw_propagators[k].state
        wrk.result.tau_vals[k] = isnothing(Ψtgt(k)) ? NaN : (Ψtgt(k) ⋅ Ψₖ)
    end
    Ψ = [p.state for p ∈ wrk.fw_propagators]
    if wrk.J_T_takes_tau
        wrk.J_parts[1] = J_T(Ψ, trajectories; tau=wrk.result.tau_vals)
    else
        wrk.J_parts[1] = J_T(Ψ, trajectories)
    end
    if !isnothing(J_a)
        wrk.J_parts[2] = λₐ * J_a(pulsevals, tlist)
    end
    return sum(wrk.J_parts)
end


"""
Evaluate the gradient ``∂J/∂ϵₙₗ`` into `G`, together with the functional `J`.

```julia
J = evaluate_gradient!(G, pulsevals, problem, wrk)
```

evaluates and returns the optimization functional defined in `problem` for the
given pulse values, cf. [`evaluate_functional`](@ref), and write the derivative
of the optimization functional with respect to the pulse values into the
existing array `G`.

The evaluation of the functional uses uses `wrk.fw_propagators`. The evaluation
of the gradient happens either via a backward propagation of an extended
["gradient vector"](@extref `QuantumGradientGenerators.GradVector`)
using `wrk.bw_grad_propagators` if `problem` was initialized with
`gradient_method=:gradgen`. Alternatively, if `problem` was initialized with
`gradient_method=:taylor`, the backward propagation if for a regular state,
using `wrk.bw_propagators`, and a Taylor expansion is used for the gradient of
the time evolution operator in a single time step.

As a side, effect, evaluating the gradient and functional sets the following
information in `wrk`:


* `wrk.pulsevals`: On output, the values of the given `pulsevals`, see
  [`evaluate_functional`](@ref).
* `wrk.result.fg_calls`: Will be incremented by one
* `wrk.fg_count[1]`: Will be incremented by one
* `wrk.result.tau_vals`: For any trajectory that defines a `target_state`, the
  overlap of the propagated state with that target state.
* `wrk.J_parts`: The parts (`J_T`, `λₐJ_a`) of the functional
* `wrk.fw_storage`: For each trajectory, the forward-propagated states at each
  point on the time grid.
* `wrk.chi_states`: The normalized states ``|χ(T)⟩`` that we used as the boundary
  condition for the backward propagation.
* `wrk.chi_states_norm`: The original norm of the states ``|χ(T)⟩``, as
  calculated by ``-∂J/∂⟨Ψₖ|``
* `wrk.grad_J_T`: The vector ``∂J_T/∂ϵ_{nl}, i.e., the gradient only for the
  final-time part of the functional
* `wrk.grad_J_a`: The vector ``∂J_a/∂ϵ_{nl}``, i.e., the gradient only for the
  pulse-dependent running cost.

The gradients are `wrk.grad_J_T` and `wrk.grad_J_a` (weighted by ``λ_a``) into
are combined into the output `G`.

Returns the value of the functional.
"""
function evaluate_gradient!(G, pulsevals, problem, wrk)

    trajectories = problem.trajectories
    N = length(trajectories)
    tlist = problem.tlist
    N_T = length(tlist) - 1  # number of time steps
    L = length(wrk.controls)

    wrk.result.fg_calls += 1
    wrk.fg_count[1] += 1

    # forward propagation and storage of states
    J_val = evaluate_functional(
        pulsevals,
        problem,
        wrk;
        storage=wrk.fw_storage,
        count_call=false
    )

    chi = wrk.kwargs[:chi]  # guaranteed to exist in `GrapeWrk` constructor
    chi_min_norm = get(problem.kwargs, :chi_min_norm, 1e-100)

    Ψ = [p.state for p ∈ wrk.fw_propagators]
    if wrk.chi_takes_tau
        # we rely on `evaluate_functional` setting the `tau_vals` as a side
        # effect
        χ = chi(Ψ, trajectories; tau=wrk.result.tau_vals)
    else
        χ = chi(Ψ, trajectories)
    end
    ρ = norm.(χ)
    χ = normalize_chis!(χ, ρ; chi_min_norm)
    wrk.chi_states = χ  # for easier debugging in a callback

    gradient_method = get(problem.kwargs, :gradient_method, :gradgen)

    if gradient_method == :gradgen

        # backward propagation of combined χ-state and gradient
        @threadsif wrk.use_threads for k = 1:N
            local Ψₖ = wrk.fw_propagators[k].state  # memory reuse
            local χ̃ₖ = GradVector(χ[k], length(wrk.controls))
            reinit_prop!(wrk.bw_grad_propagators[k], χ̃ₖ; transform_control_ranges)
            for n = N_T:-1:1  # N_T is the number of time slices
                χ̃ₖ = prop_step!(wrk.bw_grad_propagators[k])
                if haskey(wrk.bw_grad_prop_kwargs[k], :callback)
                    local cb = wrk.bw_grad_prop_kwargs[k][:callback]
                    local observables =
                        get(wrk.fw_prop_kwargs[k], :observables, _StoreState())
                    cb(wrk.bw_grad_propagators[k], observables)
                end
                if supports_inplace(Ψₖ)
                    get_from_storage!(Ψₖ, wrk.fw_storage[k], n)
                else
                    Ψₖ = get_from_storage(wrk.fw_storage[k], n)
                end
                for l = 1:L
                    wrk.tau_grads[k][n, l] = ρ[k] * (χ̃ₖ.grad_states[l] ⋅ Ψₖ)
                end
                resetgradvec!(χ̃ₖ)
                set_state!(wrk.bw_grad_propagators[k], χ̃ₖ)
            end
        end

    elseif gradient_method == :taylor

        taylor_grad_max_order = get(problem.kwargs, :taylor_grad_max_order, 100)
        taylor_grad_tolerance = get(problem.kwargs, :taylor_grad_tolerance, 1e-16)
        taylor_grad_check_convergence =
            get(problem.kwargs, :taylor_grad_check_convergence, true)

        @threadsif wrk.use_threads for k = 1:N
            local Ψₖ = wrk.fw_propagators[k].state  # memory reuse
            reinit_prop!(wrk.bw_propagators[k], χ[k]; transform_control_ranges)
            local Hₖ⁺ = wrk.adjoint_trajectories[k].generator
            local Hₖₙ⁺ = wrk.taylor_genops[k]
            for n = N_T:-1:1  # N_T is the number of time slices
                # TODO: It would be cleaner to encapsulate this in a
                # propagator-like interface that can reuse the gradgen
                # structure instead of the taylor_genops, control_derivs, and
                # taylor_grad_states in wrk
                if supports_inplace(Ψₖ)
                    get_from_storage!(Ψₖ, wrk.fw_storage[k], n)
                else
                    Ψₖ = get_from_storage(wrk.fw_storage[k], n)
                end
                for l = 1:L
                    local μₖₗ = wrk.control_derivs[k][l]
                    if isnothing(μₖₗ)
                        wrk.tau_grads[k][n, l] = 0.0
                    else
                        local ϵₙ⁽ⁱ⁾ = @view pulsevals[((n-1)*L+1):(n*L)]
                        local vals_dict = IdDict(
                            control => val for (control, val) ∈ zip(wrk.controls, ϵₙ⁽ⁱ⁾)
                        )
                        local μₗₖₙ = evaluate(μₖₗ, tlist, n; vals_dict)
                        if supports_inplace(Hₖₙ⁺)
                            evaluate!(Hₖₙ⁺, Hₖ⁺, tlist, n; vals_dict)
                        else
                            Hₖₙ⁺ = evaluate(Hₖ⁺, tlist, n; vals_dict)
                        end
                        local χₖ = wrk.bw_propagators[k].state
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
                        # TODO: taylor_grad_step for immutable states
                        wrk.tau_grads[k][n, l] = ρ[k] * dot(χ̃ₗₖ, Ψₖ)
                    end
                end
                prop_step!(wrk.bw_propagators[k])
                if haskey(wrk.bw_prop_kwargs[k], :callback)
                    local cb = wrk.bw_prop_kwargs[k][:callback]
                    local observables =
                        get(wrk.bw_prop_kwargs[k], :observables, _StoreState())
                    cb(wrk.bw_propagators[k], observables)
                end
            end
        end

    else

        error("Invalid gradient_method=$(repr(gradient_method)) ∉ (:gradgen, :taylor)")

    end

    _grad_J_T_via_chi!(wrk.grad_J_T, wrk.result.tau_vals, wrk.tau_grads)
    copyto!(G, wrk.grad_J_T)
    if haskey(wrk.kwargs, :grad_J_a)
        grad_J_a = get(wrk.kwargs, :grad_J_a, nothing)
        if !isnothing(grad_J_a)
            wrk.grad_J_a = grad_J_a(pulsevals, tlist)
            λₐ = get(wrk.kwargs, :lambda_a, 1.0)
            axpy!(λₐ, wrk.grad_J_a, G)
        end
    end
    return J_val

end


function normalize_chis!(χ::Vector{ST}, ρ::Vector{Float64}; chi_min_norm) where {ST}
    normalized_chis = Vector{ST}(undef, length(10))
    all_in_place = true
    for k in eachindex(χ)
        if ρ[k] < chi_min_norm
            error(
                "The χ state with index $k has norm $(ρ[k]) < $chi_min_norm (chi_min_norm)"
            )
        end
        if supports_inplace(χ[k])
            normalized_chis = χ
            LinearAlgebra.lmul!(1.0 / ρ[k], χ[k])
        else
            all_in_place = false
            normalized_chis[k] = χ[k] / ρ[k]
        end
    end
    if (normalized_chis ≡ χ) && (!all_in_place)
        error("Either all or none of the elements of χ must support in-place operators")
    end
    return normalized_chis
end
