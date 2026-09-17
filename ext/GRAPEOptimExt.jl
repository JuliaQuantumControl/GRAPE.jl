# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module GRAPEOptimExt

import Optim
using GRAPE: GrapeWrk, _finish_iteration!
import GRAPE: run_optimizer, step_width, search_direction


function run_optimizer(
    optimizer::Optim.AbstractOptimizer,
    wrk,
    fg!,
    callback,
    check_convergence
)

    if !(optimizer isa Optim.FirstOrderOptimizer)
        msg = "GRAPE requires a first-order Optim.jl optimizer (e.g., `Optim.LBFGS()`), not $(nameof(typeof(optimizer)))"
        throw(ArgumentError(msg))
    end
    if any(wrk.lower_bounds .> -Inf) || any(wrk.upper_bounds .< Inf)
        error("bounds are not implemented for Optim.jl optimization")
    end

    f(x) = fg!(0.0, nothing, x)
    g!(G, x) = fg!(nothing, G, x)
    fg_optim!(G, x) = fg!(0.0, G, x)
    # Note: `Optim.optimize(f, g!, fg!, …)` would interpret `fg!` as a Hessian
    objective = Optim.OnceDifferentiable(f, g!, fg_optim!, wrk.pulsevals)

    is_guess = true

    function optim_callback(state)
        # Optim.jl calls this with the optimizer state for the guess, and after
        # each iteration. At that point, `state.x` are the accepted pulse
        # values and `state.g_x` is the gradient for `state.x`. Since
        # `state.x` is always the point of the most recent call to `fg!`, all
        # "current" fields in `wrk` are for `state.x`.
        wrk.optimizer_state = state
        if wrk.pulsevals != state.x
            error("Optim.jl did not evaluate the functional for the accepted pulse values")
        end
        if is_guess
            is_guess = false
            copyto!(wrk.gradient, state.g_x)
            converged = _finish_iteration!(wrk, 0, callback, check_convergence)
        else
            iter = wrk.result.iter + 1
            converged = _finish_iteration!(wrk, iter, callback, check_convergence)
            copyto!(wrk.pulsevals_guess, wrk.pulsevals)
            copyto!(wrk.gradient, state.g_x)
        end
        if wrk.pulsevals != state.x
            error("A `callback` must not modify `pulsevals` for an Optim.jl optimizer")
        end
        return converged
    end

    options = Optim.Options(;
        callback = optim_callback,
        iterations = (wrk.result.iter_stop - wrk.result.iter_start),
        x_abstol = get(wrk.kwargs, :x_abstol, get(wrk.kwargs, :x_tol, 0.0)),
        x_reltol = get(wrk.kwargs, :x_reltol, 0.0),
        f_abstol = get(wrk.kwargs, :f_abstol, 0.0),
        f_reltol = get(wrk.kwargs, :f_reltol, get(wrk.kwargs, :f_tol, 0.0)),
        g_abstol = get(wrk.kwargs, :g_abstol, get(wrk.kwargs, :g_tol, 1e-8)),
        allow_f_increases = get(wrk.kwargs, :allow_f_increases, false),
        show_trace = get(wrk.kwargs, :show_trace, false),
        extended_trace = get(wrk.kwargs, :extended_trace, false),
        show_every = get(wrk.kwargs, :show_every, 1),
    )

    res = Optim.optimize(objective, wrk.pulsevals, optimizer, options)

    if !wrk.result.converged
        wrk.result.message = "Optim.jl terminated: $(res.termination_code)"
        @warn "Optimization failed to converge" message = wrk.result.message
    end

    return nothing

end


# Optimizers that update the pulse values as `x = x_previous + α s` for the
# `α = state.alpha` and `s = state.s` from their line search. Other optimizers
# (e.g., `ConjugateGradient`, which overwrites `state.s` with the search
# direction for the next iteration before the callback) use the generic
# `step_width` and `search_direction`.
const LineSearchOptimizer = Union{Optim.GradientDescent,Optim.BFGS,Optim.LBFGS}


function step_width(wrk::GrapeWrk{O}) where {O<:LineSearchOptimizer}
    return wrk.optimizer_state.alpha
end


function search_direction(wrk::GrapeWrk{O}) where {O<:LineSearchOptimizer}
    return wrk.optimizer_state.s
end

end
