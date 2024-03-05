import Optim

function run_optimizer(
    optimizer::Optim.AbstractOptimizer,
    wrk,
    fg!,
    info_hook,
    check_convergence!
)

    tol_options = Optim.Options(
        # just so we can instantiate `optimizer_state` before `callback`
        x_tol=get(wrk.kwargs, :x_tol, 0.0),
        f_tol=get(wrk.kwargs, :f_tol, 0.0),
        g_tol=get(wrk.kwargs, :g_tol, 1e-8),
    )
    if any(wrk.lower_bounds .> -Inf) || any(wrk.upper_bounds .< Inf)
        error("bounds are not implemented for Optim.jl optimization")
    end
    initial_x = wrk.pulsevals
    method = optimizer
    objective = Optim.promote_objtype(method, initial_x, :finite, true, Optim.only_fg!(fg!))
    optimizer_state = Optim.initial_state(method, tol_options, objective, initial_x)
    # Instantiation of `optimizer_state` calls `fg!` and sets the value of the
    # functional and gradient for the  `initial_x` in objective.F and
    # objective.DF, respectively. The `optimizer_state` is set correspondingly:
    @assert optimizer_state.x ==
            optimizer_state.x_previous ==
            objective.x_f ==
            objective.x_df
    # ... but `f_x_previous` does not match the initial `x_previous`:
    @assert isnan(optimizer_state.f_x_previous)

    # update the result object and check convergence
    function callback(optimization_state::Optim.OptimizationState)
        iter = wrk.result.iter_start + optimization_state.iteration
        #@assert optimization_state.value == objective.F
        #if optimization_state.iteration > 0
        #    @assert norm(
        #       optimizer_state.x .-
        #       (optimizer_state.x_previous .+ optimizer_state.alpha .* optimizer_state.s)
        #    ) < 1e-14
        #end
        if hasproperty(optimizer_state, :s) && hasproperty(optimizer_state, :alpha)
            wrk.searchdirection .= optimizer_state.s
            wrk.alpha = optimizer_state.alpha
        elseif (optimization_state.iteration == 1)
            @error "Cannot determine search direction/step width"
        end
        update_result!(wrk, iter)
        #update_hook!(...) # TODO
        info_tuple = info_hook(wrk, wrk.result.iter)
        if hasproperty(objective, :DF)
            # DF is the *current* gradient, i.e., the gradient of the updated
            # pulsevals, which (after the call to `info_hook`) is the gradient
            # for the the guess of the next iteration.
            wrk.gradient .= objective.DF
        elseif (optimization_state.iteration == 1)
            @error "Cannot determine guess gradient"
        end
        copyto!(wrk.pulsevals_guess, wrk.pulsevals)
        wrk.fg_count .= 0
        (info_tuple !== nothing) && push!(wrk.result.records, info_tuple)
        check_convergence!(wrk.result)
        return wrk.result.converged
    end

    options = Optim.Options(
        callback=callback,
        iterations=wrk.result.iter_stop - wrk.result.iter_start, # TODO
        x_tol=get(wrk.kwargs, :x_tol, 0.0),
        f_tol=get(wrk.kwargs, :f_tol, 0.0),
        g_tol=get(wrk.kwargs, :g_tol, 1e-8),
        show_trace=get(wrk.kwargs, :show_trace, false),
        extended_trace=get(wrk.kwargs, :extended_trace, false),
        store_trace=get(wrk.kwargs, :store_trace, false),
        show_every=get(wrk.kwargs, :show_every, 1),
        allow_f_increases=get(wrk.kwargs, :allow_f_increases, false),
    )

    res = Optim.optimize(objective, initial_x, method, options, optimizer_state)

    if !res.ls_success
        @error "optimization failed (linesearch)"
        wrk.result.message = "Failed linesearch"
    end
    if res.stopped_by.f_increased
        @error "loss of monotonic convergence (try allow_f_increases=true)"
        wrk.result.message = "Loss of monotonic convergence"
    end
    if !wrk.result.converged
        @warn "Optimization failed to converge"
    end

    return nothing

end
