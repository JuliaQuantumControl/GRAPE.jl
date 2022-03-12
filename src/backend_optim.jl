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
    @assert optimizer_state.g_previous == objective.DF
    # ... but `f_x_previous` does not match the initial `x_previous`:
    @assert isnan(optimizer_state.f_x_previous)

    # update the result object and check convergence
    function callback(optimization_state::Optim.OptimizationState)
        @assert optimization_state.value == objective.F
        #if optimization_state.iteration > 0
        #    @assert norm(
        #       optimizer_state.x .-
        #       (optimizer_state.x_previous .+ optimizer_state.alpha .* optimizer_state.s)
        #    ) < 1e-14
        #end
        wrk.gradient .= optimizer_state.g_previous
        wrk.searchdirection .= optimizer_state.s  # TODO: may depend on type of optimizer_state
        iter = wrk.result.iter_start + optimization_state.iteration
        update_result!(wrk, optimization_state, optimizer_state, iter)
        #update_hook!(...) # TODO
        info_tuple = info_hook(wrk, optimization_state, optimizer_state, wrk.result.iter)
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
    return res

end


function update_result!(
    wrk::GrapeWrk,
    optimization_state::Optim.OptimizationState,
    optimizer_state::Optim.AbstractOptimizerState,
    i::Int64
)
    # TODO: make this depend only on wrk. Should not be backend-dependent
    res = wrk.result
    for (k, Ψ) in enumerate(wrk.fw_states)
        copyto!(res.states[k], Ψ)
    end
    res.J_T_prev = res.J_T
    res.J_T = optimization_state.value
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
    for l = 1:L
        res.optimized_controls[l] = discretize(ϵ_opt[l, :], res.tlist)
    end
    res.optim_res = optim_res
end


"""Print optimization progress as a table.

This functions serves as the default `info_hook` for an optimization with
GRAPE.
"""
function print_table(
    wrk,
    optimization_state::Optim.OptimizationState,
    optimizer_state::Optim.AbstractOptimizerState,
    iteration,
    args...
)
    # TODO: make this depend only on wrk. Should not be backend-dependent
    J_T = wrk.result.J_T
    ΔJ_T = J_T - wrk.result.J_T_prev
    secs = wrk.result.secs

    iter_stop = "$(get(wrk.kwargs, :iter_stop, 5000))"
    widths = [max(length("$iter_stop"), 6), 11, 11, 11, 8, 8]

    if iteration == 0
        header = ["iter.", "J_T", "|∇J_T|", "ΔJ_T", "FG(F)", "secs"]
        for (header, w) in zip(header, widths)
            print(lpad(header, w))
        end
        print("\n")
    end

    strs = (
        "$iteration",
        @sprintf("%.2e", J_T),
        @sprintf("%.2e", optimization_state.g_norm),
        (iteration > 0) ? @sprintf("%.2e", ΔJ_T) : "n/a",
        @sprintf("%d(%d)", wrk.fg_count[1], wrk.fg_count[2]),
        @sprintf("%.1f", secs),
    )
    for (str, w) in zip(strs, widths)
        print(lpad(str, w))
    end
    print("\n")
end
