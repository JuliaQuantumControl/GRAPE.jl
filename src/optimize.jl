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

# BEGIN DEBUG

DEBUG_FH = nothing

function open_debug_file(filename)
    global DEBUG_FH
    DEBUG_FH = open(filename, "w")
end

function close_debug_file()
    global DEBUG_FH
    close(DEBUG_FH)
    DEBUG_FH = nothing
end

function _c(c; fmt=:float)
    a = real(c)
    b = imag(c)
    if abs(a) < 1e-10
        a = 0.0
    end
    if abs(b) < 1e-10
        b = 0.0
    end
    if fmt == :float
        return @sprintf("%+.4f%+.4f𝕚", a, b)
    elseif fmt == :exp
        return @sprintf("%+.2e%+.2e𝕚", a, b)
    else
        throw(ArgumentError("Invalid fmt=$(repr(fmt))"))
    end
end

function _state(Ψ; fmt=:float)
    return "(" * join([_c(c; fmt) for c::ComplexF64 in Ψ], ",") * ")"
end
# END DEBUG

import QuantumControl: optimize, make_print_iters

@doc raw"""
```julia
using GRAPE
result = optimize(problem; method=GRAPE, kwargs...)
```

optimizes the given control [`problem`](@ref QuantumControl.ControlProblem)
via the GRAPE method, by minimizing the functional

```math
J(\{ϵ_{nl}\}) = J_T(\{|ϕ_k(T)⟩\}) + λ_a J_a(\{ϵ_{nl}\})
```

where the final time functional ``J_T`` depends explicitly on the
forward-propagated states and the running cost ``J_a`` depends explicitly on
pulse values ``ϵ_{nl}`` of the l'th control discretized on the n'th interval of
the time grid.

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
  from `J_T` via [`make_chi`](@ref) with the default parameters. Similarly to
  `J_T`, if `chi` accepts a keyword argument `tau`, it will be passed a vector
  of complex overlaps.
* `J_a`: A function `J_a(pulsevals, tlist)` that evaluates running costs over
  the pulse values, where `pulsevals` are the vectorized values ``ϵ_{nl}``,
  where `n` are in indices of the time intervals and `l` are the indices over
  the controls, i.e., `[ϵ₁₁, ϵ₂₁, …, ϵ₁₂, ϵ₂₂, …]` (the pulse values for each
  control are contiguous). If not given, the optimization will not include a
  running cost.
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
* `store_iter_info=Set()`: Which fields from `print_iters` to store in
  `result.records`. A subset of
  `Set(["iter.", "J_T", "|∇J_T|", "ΔJ_T", "FG(F)", "secs"])`.
* `callback`: A function (or tuple of functions) that receives the
  [GRAPE workspace](@ref GrapeWrk) and the iteration number. The function
  may return a tuple of values which are stored in the
  [`GrapeResult`](@ref) object `result.records`. The function can also mutate
  the workspace, in particular the updated `pulsevals`. This may be used,
  e.g., to apply a spectral filter to the updated pulses or to perform
  similar manipulations. Note that `print_iters=true` (default) adds an
  automatic callback to print information after each iteration. With
  `store_iter_info`, that callback automatically stores a subset of the
  printed information.
* `check_convergence`: A function to check whether convergence has been
  reached. Receives a [`GrapeResult`](@ref) object `result`, and should set
  `result.converged` to `true` and `result.message` to an appropriate string in
  case of convergence. Multiple convergence checks can be performed by chaining
  functions with `∘`. The convergence check is performed after any `callback`.
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
* `prop_method`: The propagation method to use for each trajectory, see below.
* `verbose=false`: If `true`, print information during initialization
* `rethrow_exceptions`: By default, any exception ends the optimization, but
  still returns a [`GrapeResult`](@ref) that captures the message associated
  with the exception. This is to avoid losing results from a long-running
  optimization when an exception occurs in a later iteration. If
  `rethrow_exceptions=true`, instead of capturing the exception, it will be
  thrown normally.

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
    gradient_method = get(problem.kwargs, :gradient_method, :gradgen)
    taylor_grad_max_order = get(problem.kwargs, :taylor_grad_max_order, 100)
    taylor_grad_tolerance = get(problem.kwargs, :taylor_grad_tolerance, 1e-16)
    taylor_grad_check_convergence =
        get(problem.kwargs, :taylor_grad_check_convergence, true)

    wrk = GrapeWrk(problem; verbose)

    Ψ₀ = [traj.initial_state for traj ∈ wrk.trajectories]
    Ψtgt = Union{eltype(Ψ₀),Nothing}[
        (hasproperty(traj, :target_state) ? traj.target_state : nothing) for
        traj ∈ wrk.trajectories
    ]

    J = wrk.J_parts
    tlist = wrk.result.tlist
    J_T = wrk.kwargs[:J_T]
    J_a_func = get(wrk.kwargs, :J_a, nothing)
    ∇J_T = wrk.grad_J_T
    λₐ = get(wrk.kwargs, :lambda_a, 1.0)
    chi = wrk.kwargs[:chi]  # guaranteed to exist in `GrapeWrk` constructor
    grad_J_a = nothing
    if !isnothing(J_a_func)
        if haskey(wrk.kwargs, :grad_J_a)
            grad_J_a = wrk.kwargs[:grad_J_a]
        else
            # With a manually given `grad_J_a`, the `make_grad_J_a` function
            # should never be called. So we can't use `get` to set this.
            grad_J_a = make_grad_J_a(J_a_func, tlist)
        end
    end

    τ = wrk.result.tau_vals
    ∇τ = wrk.tau_grads
    N_T = length(tlist) - 1
    N = length(wrk.trajectories)
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
                if haskey(wrk.fw_prop_kwargs[k], :callback)
                    local cb = wrk.fw_prop_kwargs[k][:callback]
                    local observables =
                        get(wrk.fw_prop_kwargs[k], :observables, _StoreState())
                    cb(wrk.fw_propagators[k], observables)
                end
                (Φₖ !== nothing) && write_to_storage!(Φₖ, n + 1, Ψₖ)
            end
            local Ψₖ = wrk.fw_propagators[k].state
            τ[k] = isnothing(Ψtgt[k]) ? NaN : (Ψtgt[k] ⋅ Ψₖ)
        end
        Ψ = [p.state for p ∈ wrk.fw_propagators]
        if wrk.J_T_takes_tau
            J[1] = J_T(Ψ, wrk.trajectories; tau=τ)
        else
            J[1] = J_T(Ψ, wrk.trajectories)
        end
        if !isnothing(J_a_func)
            J[2] = λₐ * J_a_func(pulsevals, tlist)
        end
        return sum(J)
    end

    # Calculate the functional and the gradient G ≡ ∇J_T
    # Side-effects:
    # as in f(...); wrk.grad_J_T, wrk.grad_J_a
    function fg_gradgen!(F, G, pulsevals)

        if !isnothing(DEBUG_FH)  # DEBUG
            println(DEBUG_FH, "# fg_gradgen! in iter $(wrk.result.iter)")
        end
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
        if wrk.chi_takes_tau
            χ = chi(Ψ, wrk.trajectories; tau=τ)  # τ is set in f()
        else
            χ = chi(Ψ, wrk.trajectories)
        end
        wrk.chi_states = χ  # for easier debugging in a callback
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
                    get_from_storage!(Ψₖ, Φ[k], n)
                else
                    Ψₖ = get_from_storage(Φ[k], n)
                end
                for l = 1:L
                    ∇τ[k][n, l] = χ̃ₖ.grad_states[l] ⋅ Ψₖ
                    if !isnothing(DEBUG_FH) && (k == 1) && (l == 1) # DEBUG
                        msg = "n = $(@sprintf("%04d", n)), Ψₖ=$(_state(Ψₖ)), χₖ=$(_state(χ̃ₖ.state)), χ̃ₗₖ=$(_state(χ̃ₖ.grad_states[l]; fmt=:exp)) ⇒ ∇τₙ = $(_c(∇τ[k][n, l]; fmt=:exp))"
                        println(DEBUG_FH, msg)
                    end
                end
                resetgradvec!(χ̃ₖ)
                set_state!(wrk.bw_grad_propagators[k], χ̃ₖ)
            end
        end

        _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
        copyto!(G, ∇J_T)
        if !isnothing(grad_J_a)
            wrk.grad_J_a = grad_J_a(pulsevals, tlist)
            axpy!(λₐ, wrk.grad_J_a, G)
        end
        return J_val

    end

    # Calculate the functional and the gradient G ≡ ∇J_T
    # Side-effects:
    # as in f(...); wrk.grad_J_T, wrk.grad_J_a
    function fg_taylor!(F, G, pulsevals)

        if !isnothing(DEBUG_FH)  # DEBUG
            println(DEBUG_FH, "# fg_taylor! in iter $(wrk.result.iter)")
        end
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
        if wrk.chi_takes_tau
            χ = chi(Ψ, wrk.trajectories; tau=τ)  # τ is set in f()
        else
            χ = chi(Ψ, wrk.trajectories)
        end
        wrk.chi_states = χ  # for easier debugging in a callback
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
                if ismutable(Ψₖ)
                    get_from_storage!(Ψₖ, Φ[k], n)
                else
                    Ψₖ = get_from_storage(Φ[k], n)
                end
                for l = 1:L
                    local μₖₗ = wrk.control_derivs[k][l]
                    if isnothing(μₖₗ)
                        ∇τ[k][n, l] = 0.0
                    else
                        local ϵₙ⁽ⁱ⁾ = @view pulsevals[(n-1)*L+1:n*L]
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
                        ∇τ[k][n, l] = dot(χ̃ₗₖ, Ψₖ)
                        if !isnothing(DEBUG_FH) && (k == 1) && (l == 1) # DEBUG
                            msg = "n = $(@sprintf("%04d", n)), Ψₖ=$(_state(Ψₖ)), χₖ=$(_state(χₖ)), χ̃ₗₖ=$(_state(χ̃ₗₖ; fmt=:exp)) ⇒ ∇τₙ = $(_c(∇τ[k][n, l]; fmt=:exp))"
                            println(DEBUG_FH, msg)
                        end
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

        _grad_J_T_via_chi!(∇J_T, τ, ∇τ)
        copyto!(G, ∇J_T)
        if !isnothing(grad_J_a)
            wrk.grad_J_a = grad_J_a(pulsevals, tlist)
            axpy!(λₐ, wrk.grad_J_a, G)
        end
        return J_val

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
        if gradient_method == :gradgen
            run_optimizer(optimizer, wrk, fg_gradgen!, callback, check_convergence!)
        elseif gradient_method == :taylor
            run_optimizer(optimizer, wrk, fg_taylor!, callback, check_convergence!)
        else
            error("Invalid gradient_method=$(repr(gradient_method)) ∉ (:gradgen, :taylor)")
        end
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


function run_optimizer end


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


function finalize_result!(wrk::GrapeWrk)
    L = length(wrk.controls)
    res = wrk.result
    res.end_local_time = now()
    N_T = length(res.tlist) - 1
    for l = 1:L
        ϵ_opt = wrk.pulsevals[(l-1)*N_T+1:l*N_T]
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

    headers = ["iter.", "J_T", "|∇J_T|", "ΔJ_T", "FG(F)", "secs"]
    store_iter_info = Set(get(kwargs, :store_iter_info, Set()))
    info_vals = Vector{Any}(undef, length(headers))
    fill!(info_vals, nothing)
    store_iter = false
    store_J_T = false
    store_grad_norm = false
    store_ΔJ_T = false
    store_counts = false
    store_secs = false
    for item in store_iter_info
        if item == "iter."
            store_iter = true
        elseif item == "J_T"
            store_J_T = true
        elseif item == "|∇J_T|"
            store_grad_norm = true
        elseif item == "ΔJ_T"
            store_ΔJ_T = true
        elseif item == "FG(F)"
            store_counts = true
        elseif item == "secs"
            store_secs = true
        else
            msg = "Item $(repr(item)) in `store_iter_info` is not one of $(repr(headers)))"
            throw(ArgumentError(msg))
        end
    end

    function print_table(wrk, iteration, args...)

        J_T = wrk.result.J_T
        ΔJ_T = J_T - wrk.result.J_T_prev
        secs = wrk.result.secs
        grad_norm = norm(wrk.grad_J_T)
        counts = Tuple(wrk.fg_count)

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

        store_iter && (info_vals[1] = iteration)
        store_J_T && (info_vals[2] = J_T)
        store_grad_norm && (info_vals[3] = grad_norm)
        store_ΔJ_T && (info_vals[4] = ΔJ_T)
        store_counts && (info_vals[5] = counts)
        store_secs && (info_vals[6] = secs)

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
            @sprintf("%.2e", grad_norm),
            (iteration > 0) ? @sprintf("%.2e", ΔJ_T) : "n/a",
            @sprintf("%d(%d)", counts[1], counts[2]),
            @sprintf("%.1f", secs),
        ]
        for (str, header) in zip(strs, headers)
            w = width[header]
            print(lpad(str, w))
        end
        print("\n")
        flush(stdout)

        return Tuple((value for value in info_vals if (value !== nothing)))

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
