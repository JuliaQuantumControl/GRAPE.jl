using QuantumControlBase
using QuantumPropagators
using Parameters
using ConcreteStructs

@concrete struct GrapeWrk
    objectives # copy of objectives
    pulse_mapping # as michael describes, similar to c_ops
    H_store # store for Hamiltonian
    ψ_store # store for forward states
    ϕ_store # store for forward states
    aux_state # store for [psi 0]
    aux_store # store for auxiliary matrix
    dP_du # store for directional derivative
    tlist # tlist 
    prop_wrk # prop wrk
    aux_prop_wrk # aux prop wrk
end

function GrapeWrk(objectives, tlist, prop_method, pulse_mapping = "")
    N_obj = length(objectives)

    @unpack initial_state, generator, target_state = objectives[1]

    ψ = initial_state

    controls = getcontrols(generator)
    N_slices = length(tlist)
    dim = size(initial_state, 1)
    # store forward evolution
    ψ_store = [[similar(initial_state) for i = 1:N_slices] for ii = 1:N_obj]

    for i = 1:N_obj
        ψ_store[i][1] = objectives[i].initial_state
    end

    ϕ_store = [[similar(initial_state) for i = 1:N_slices] for ii = 1:N_obj]

    for i = 1:N_obj
        ϕ_store[i][N_slices] = objectives[i].target_state
    end

    H_store = [[similar(generator[1]) for i = 1:N_slices] for ii = 1:N_obj]

    aux_mat = [zeros(eltype(initial_state), 2 * dim, 2 * dim) for i = 1:N_obj]
    aux_state = [zeros(eltype(initial_state), 2 * dim) for i = 1:N_obj]

    dP_du = [
        [[zeros(eltype(ψ), size(ψ)) for i = 1:N_slices] for k = 1:length(controls)] for
        ii = 1:N_obj
    ]

    prop_wrk = [initpropwrk(obj.initial_state, tlist; prop_method) for obj in objectives]

    aux_prop_wrk = [initpropwrk(aux_state[1], tlist; prop_method) for obj in objectives]
    return GrapeWrk(
        objectives,
        pulse_mapping,
        H_store,
        ψ_store,
        ϕ_store,
        aux_state,
        aux_mat,
        dP_du,
        tlist,
        prop_wrk,
        aux_prop_wrk,
    )
end

function optimize(wrk, pulse_options, tlist, propagator)
    @unpack objectives,
    pulse_mapping,
    H_store,
    ψ_store,
    ϕ_store,
    aux_state,
    aux_store,
    dP_du,
    tlist,
    prop_wrk,
    aux_prop_wrk = wrk

    controls = getcontrols(objectives)
    pulses = [discretize_on_midpoints(control, tlist) for control in controls]

    N_obj = length(objectives)
    N_slices = length(tlist) - 1
    N_controls = size(controls, 1)
    dim = size(H_store[1][1], 1)
    dt = tlist[2] - tlist[1]
    obj = 1
    # now we need to make a fn of F, G, x
    function test_grape(
        F,
        G,
        x,
        dim,
        ψ_store,
        ϕ_store,
        temp_state,
        aux_store,
        dd_store,
        grad,
    )
        @inbounds for n = 1:N_slices
            # save the initial state in each timestep
            ψ_not_mutated = copy(ψ_store[obj][n])
            # copy the state into the next slice since propstep! will mutate it
            ψ_store[obj][n+1] .= ψ_store[obj][n]
            # save the Hamiltonian for computation later
            H_store[obj][n] .= H[1] + H[2][1] .* x[1][n]
            ψ = wrk.ψ_store[obj][n+1]
            propstep!(ψ, H_store[obj][n], dt, prop_wrk[obj])

            aux_state[obj][dim+1:end] .= ψ_not_mutated

            temp_state[dim+1:end] .= ψ
            @inbounds for k = 1:N_controls
                # will tidy up using a struct later
                aux_store[obj][1:dim, dim+1:end] .= H[2][1]
                aux_store[obj][1:dim, 1:dim] .= H_store[obj][n]
                aux_store[obj][dim+1:end, dim+1:end] .= H_store[obj][n]
                # propagate the auxilliary state forwards in time using the auxilliary matrix
                propstep!(aux_state[obj], aux_store[obj], dt, aux_prop_wrk[obj])
                # we only save the small part that we care about
                dP_du[obj][k][n] .= aux_state[obj][1:dim]
                aux_state[obj] .= 0.0 + 0.0im
            end
        end

        # @show abs2.(ψ_store[end])

        for n in reverse(1:N_slices)
            ϕ = ϕ_store[n+1]
            ϕ_store[n] .= expv(1.0im * dt, H_store[n], ϕ)
        end

        fid = abs2(ϕ_store[N]' * ψ_store[N])
        for n = 1:N
            for k = 1:K
                grad[k, n] =
                    2 * real(ϕ_store[n]' * dd_store[k][n] * ψ_store[n]' * ϕ_store[N])
            end
        end

        if G !== nothing
            G .= grad
        end
        if F !== nothing
            return fid
        end


    end

end
