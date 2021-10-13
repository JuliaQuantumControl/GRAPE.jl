using QuantumControlBase
using QuantumPropagators
using Parameters
using ConcreteStructs
using Optim

import LinearAlgebra


@concrete struct GrapeWrk
    objectives # copy of objectives
    pulse_mapping # as michael describes, similar to c_ops
    N_slices # number of slices because its nice to store
    ϕ_store # store for forward states
    G # store for the generator
    vals_dict # dictionary that will store the array mapping at each point in time
    tlist # tlist
    prop_wrk # prop wrk
    aux_prop_wrk # aux prop wrk
    controls
    td_gradgens # time dependent grad generators, one for each objective
    gradvecs # gradvecs for each objective
end

function GrapeWrk(objectives, tlist, prop_method, pulse_mapping = "")
    N_obj = length(objectives)
    @unpack initial_state, generator, target_state = objectives[1]
    # copy from Krotov
    controls = getcontrols(generator)

    N_controls = size(controls, 1)

    pulses0 = [discretize_on_midpoints(control, tlist) for control in controls]

    N_slices = size(pulses0[1], 1)

    zero_vals =
        IdDict(control => zero(pulses0[i][1]) for (i, control) in enumerate(controls))
    G = [evalcontrols(obj.generator, zero_vals) for obj in objectives]
    vals_dict = [copy(zero_vals) for _ in objectives]

    # store for the backward evolution
    ϕ_store = [[similar(initial_state) for i = 1:N_slices+1] for ii = 1:N_obj]
    # similarly we set the final entry of each to the target state
    for i = 1:N_obj
        ϕ_store[i][N_slices+1] .= copy(objectives[i].target_state)
    end

    td_gradgens = [TimeDependentGradGenerator(obj.generator) for obj in objectives]

    # propagator working structs, many for each objective
    prop_wrk = [initpropwrk(obj.initial_state, tlist; prop_method) for obj in objectives]

    Ψ_grad = GradVector(copy(initial_state), N_controls)
    aux_prop = [initpropwrk(Ψ_grad, tlist, :newton) for _ in objectives]

    gradvecs = [GradVector(Ψ_store[i][1], N_controls) for i = 1:N_obj]

    return GrapeWrk(
        objectives,
        pulse_mapping,
        N_slices,
        ϕ_store,
        G,
        vals_dict,
        tlist,
        prop_wrk,
        aux_prop,
        controls,
        td_gradgens,
        gradvecs,
    )
end

function optimize(wrk, pulse_options)
    @unpack objectives,
    pulse_mapping,
    N_slices,
    Ψ_store,
    ϕ_store,
    G,
    vals_dict,
    dP_du,
    tlist,
    prop_wrk,
    controls,
    td_gradgens = wrk

    N_obj = length(objectives)
    N_controls = size(controls, 1)
    dim = size(Ψ_store[1][1], 1)

    function grape_all_obj(
        F,
        G,
        x,
        N_obj,
        N_slices,
        N_controls,
        Ψ_store,
        ϕ_store,
        grad,
        wrk,
        prop_wrk,
        td_gradgens,
        gradvec,
    )
        # in the cases where we want to use these just ensure that they're 0
        if F !== nothing
            F = 0.0
        end
        if G !== nothing
            G .= G .* 0.0
        end
        # for each objective
        # we can do this in parallel
        @inbounds for obj = 1:N_obj
            timedepgradgen = td_gradgens[obj]
            Ψ_grad = gradvec[obj]

            _bw_prop!(x, ϕ_store[obj], N_slices, obj, wrk, prop_wrk[obj])

            @inbounds for n = 1:N_slices
                # reset the gradvec so its ready for mutation during evolution
                resetgradvec!(Ψ_grad, Ψ_store[obj][n])
                # get the vals dict at this timeslice
                vals_dict = _get_vals_dict(x[:, n], obj, n, wrk)
                dt = tlist[n+1] - tlist[n]
                # and then get the grad generator
                grad_generator = evalcontrols(timedepgradgen, vals_dict)
                # propagate the state and the gradient forward in time
                propstep!(Ψ_grad, grad_generator, dt, aux_prop_wrk[obj])

                # compute the gradient
                for ctrl = 1:N_controls
                    grad[ctrl, n] = 2 * imag(ϕ_store[obj][n]' * Ψ_grad.grad_states[ctrl])
                end

            end
            # compute the final fidelity
            τ = real(abs2(ϕ_store[obj][N_slices+1]' * Ψ_grad.state))
            # add scaled gradient
            if G !== nothing
                G .= +grad / N_obj
            end
            # sum figure of merit, needs weights
            if F !== nothing
                F = F + τ / N_obj
            end

        end

    end

    # closure over the variables so that we can optimise
    topt =
        (F, G, x) -> grape_all_obj(
            F,
            G,
            x,
            N_obj,
            N_slices,
            N_controls,
            Ψ_store,
            ϕ_store,
            grad,
            wrk,
            prop_wrk,
            td_gradgens,
            gradvec,
        )
    # we minimize the result and return
    minimize(Optim.onlyfg!(topt), pulses, LBFGS())


end



"""
Evaluate the total Generator (composed of static and ϵ*dynamic) at a time index [n] and at objective index [k]
"""
function _eval_gen(ϵ, k, n, wrk)
    vals_dict = _get_vals_dict(ϵ, k, n, wrk)
    tlist = wrk.tlist
    # will this ever go out of bounds? TODO check
    dt = tlist[n+1] - tlist[n]
    evalcontrols!(wrk.G[k], wrk.objectives[k].generator, vals_dict)
    return wrk.G[k], dt
end

function _get_vals_dict(ϵ, k, n, wrk)
    vals_dict = wrk.vals_dict[k]
    t = wrk.tlist
    for (l, control) in enumerate(wrk.controls)
        vals_dict[control] = ϵ[l][n]
    end
    vals_dict
end


"""
Backwards propagate the states ϕ and store them, this will propagate over every single timestep
"""
function _bw_prop!(x, ϕ_store, N_slices, k_ens, grapewrk, prop_wrk)
    @inbounds for n in reverse(1:N_slices)
        ϕ_store[n] .= ϕ_store[n+1]
        G, dt = _eval_gen(x, k_ens, n, grapewrk)
        propstep!(ϕ_store[n], G, -1.0 * dt, prop_wrk)
    end
end
