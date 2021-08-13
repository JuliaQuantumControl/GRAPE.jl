using QuantumControlBase
using Parameters
using ConcreteStructs

@concrete struct GRAPEWrk
    objectives # copy of objectives
    pulse_mapping # as michael describes, similar to c_ops
    H_store # store for Hamiltonian
    ψ_store # store for forward states
    ϕ_store # store for reverse states
    temp_state # store for [psi 0]
    aux_store # store for auxiliary matrix
    dd_store # store for directional derivative
end

function GRAPEWrk(objective, n_slices, n_controls; pulse_mapping="")
    @unpack initial_state, H, target = objective
    dim, ψ_store, ϕ_store, temp_state, aux_mat, dd_store, H_store = _get_storage_arrays(initial_state, target, n_slices, n_controls)

    return GRAPEWrk(objective, pulse_mapping, H_store, ψ_store, ϕ_store, temp_state, aux_mat, dd_store)

end

function _get_storage_arrays(ψ, ϕ, N_slices, K_controls)
    dim = size(ψ,1)
    # you will need a store for both states and costates
    ψ_store = [zeros(eltype(ψ), size(ψ)) for i = 1:N_slices+1]
    ϕ_store = [zeros(eltype(ϕ), size(ϕ)) for i = 1:N_slices+1]
    ψ_store[1] .= ψ
    ϕ_store[N_slices+1] .= ϕ
    # will need zeros for the state
    zerofs = zero(ψ)
    # and a temp state
    temp_state = [zerofs; zerofs]
    # and for the internals of the aux matrix
    zeromat = zeros(eltype(ψ), (dim, dim))
    # you also need somewhere to store the auxil matrix
    aux_mat = [zeromat zeromat; zeromat zeromat]
    # finally somewhere to store the directional derivative vector
    directderiv_store = [[zeros(eltype(ψ), size(ψ)) for i = 1:N_slices] for k = 1:K_controls]
    H_store = [zeros(eltype(ψ), (dim, dim)) for i = 1:N_slices]
    return dim, ψ_store, ϕ_store, temp_state, aux_mat, directderiv_store, H_store
end

function optimize()

end
