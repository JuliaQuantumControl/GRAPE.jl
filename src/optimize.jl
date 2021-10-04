using QuantumControlBase
using QuantumPropagators
using Parameters
using ConcreteStructs
using Optim

@concrete struct GrapeWrk
    objectives::Any # copy of objectives
    pulse_mapping::Any # as michael describes, similar to c_ops
    H_store::Any # store for Hamiltonian
    ψ_store::Any # store for forward states
    ϕ_store::Any # store for forward states
    aux_state::Any # store for [psi 0]
    aux_store::Any # store for auxiliary matrix
    dP_du::Any # store for directional derivative
    tlist::Any # tlist 
    prop_wrk::Any # prop wrk
    aux_prop_wrk::Any # aux prop wrk
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
    grad = [similar(pulse) for pulse in pulses]
    N_obj = length(objectives)
    N_slices = length(tlist) - 1
    N_controls = size(controls, 1)
    dim = size(H_store[1][1], 1)
    dt = tlist[2] - tlist[1]

    function grape_all_obj(
        F,
        G,
        x,
        N_obj,
        N_slices,
        N_controls,
        dim,
        ψ_store,
        ϕ_store,
        H_store,
        aux_state,
        aux_store,
        dP_du,
        dt,
        grad,
    )
        # in the cases where we want to use these just ensure that they're 0
        if F !== nothing
            F = 0.0
        end
        if G !== nothing
            G .= G .* 0.0
        end
        # for each objective
        @inbounds for obj = 1:N_obj
            # forward prop all states for current objective
            ψ_store_copy = copy(ψ_store[obj])
            _fw_prop!(x, ψ_store[obj], H_store[obj], N_slices, dt, prop_wrk[obj], H)
            _fw_prop_aux!(
                aux_state[obj],
                aux_store[obj],
                dim,
                H[2][:],
                H_store[obj],
                N_controls,
                N_slices,
                dt,
                ψ_store_copy,
                aux_prop_wrk[obj],
                dP_du[obj],
            )

            _bw_prop!(ϕ_store[obj], H_store[obj], N_slices, dt, prop_wrk[obj])

            fid = abs2(ϕ_store[obj][N_slices]' * ψ_store[obj][N_slices])

            @inbounds for n = 1:N_slices
                @inbounds for k = 1:N_controls
                    grad[k][n] = 2 * real(ϕ_store[obj][n]' * dP_du[obj][k][n])
                end
            end


            if G !== nothing
                G .= +grad / N_obj
            end
            if F !== nothing
                F = F + fid / N_obj
            end

        end

    end


    topt =
        (F, G, x) -> grape_all_obj(
            F,
            G,
            x,
            N_obj,
            N_slices,
            N_controls,
            dim,
            ψ_store,
            ϕ_store,
            H_store,
            aux_state,
            aux_store,
            dP_du,
            dt,
            grad,
        )

    minimize(Optim.onlyfg!(topt), pulses, LBFGS())


end


"""
Propagate system forward in time and store states at a constant objective index
"""
function _fw_prop!(x, ψ_store, H_store, N_slices, dt, prop_wrk, H)
    @inbounds for i = 1:N_slices
        ψ_store[n+1] .= ψ_store[n]
        H_store[n] .= H[1] + H[2][1] .* x[1][n]
        ψ = ψ_store[n+1]
        propstep!(ψ, H_store[n], dt, prop_wrk)
    end
end

"""
Propagate auxilliary system forward in time and then store the states at a constant objective index
"""
function _fw_prop_aux!(
    aux_state,
    aux_store,
    dim,
    Hc,
    H_store,
    N_controls,
    N_slices,
    dt,
    ψ_store,
    aux_prop_wrk,
    dP_du,
)
    @inbounds for i = 1:N_slices
        aux_state[obj][dim+1:end] .= ψ_store[n]
        @inbounds for k = 1:N_controls
            aux_store[1:dim, dim+1:end] .= Hc
            aux_store[1:dim, 1:dim] .= H_store[n]
            aux_store[dim+1:end, dim+1:end] = H_store[n]
            propstep!(aux_state, aux_store, dt, aux_prop_wrk)

            dP_du[k][n] .= aux_state[1:dim]
            aux_state[obj] .= 0.0 + 0.0im
        end
    end
end

"""
Backwards propagate the states ϕ and store them
"""
function _bw_prop!(ϕ_store, H_store, N_slices, dt, prop_wrk)
    @inbounds for n in reverse(1:N_slices)
        ϕ_store[n] .= ϕ_store[n+1]
        ϕ = ϕ_store[n]
        propstep!(ϕ, H_store[n], -1.0 * dt, prop_wrk)
    end
end
