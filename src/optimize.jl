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

function optimize(wrk, pulse_options, tlist, propagator, )
    @unpack objectives, pulse_mapping, H_store, ψ_store, ϕ_store, temp_state,aux_store, dd_store = wrk

    # now we need to make a fn of F, G, x
    function test_grape(F, G, x, dim, ψ_store, ϕ_store, temp_state, aux_store, dd_store, grad, H_func, H_K_super)
        for n = 1:N
            ψ = ψ_store[n]
            # could compute Hamiltonian once and store it
            H_store[n] .= H_func(x, n * dt, T, N)
            
            # U * ψ
            ψ_store[n+1] .= expv(-1.0im * dt, H_store[n], ψ)
        
            # compute and store the directional derivative
            # set up the temp_state 
            temp_state[dim+1:end] .= ψ
            for k = 1:K
                # set up augmented matrix
                H_k = H_K_super[k]
                aux_store[1:dim, dim+1:end] .= H_k
                # then on the diagonal we want H_store[n]
                aux_store[1:dim, 1:dim] .= H_store[n]
                aux_store[dim+1:end, dim+1:end] .= H_store[n]
                dd_store[k][n] .= expv(-1.0im * dt, aux_store, temp_state)[1:dim]
            end
        end
    
        # @show abs2.(ψ_store[end])
    
        for n = reverse(1:N)
            ϕ = ϕ_store[n+1]
            # H_n_temp = H_func(x, n * dt, T, N)
            # id = I(size(H_n_temp,1))
            
            # H_n = kron(id, H_n_temp) - kron(H_n_temp', id)
            ϕ_store[n] .= expv(1.0im * dt, H_store[n], ϕ)
        end
    
        fid = abs2(ϕ_store[N]' * ψ_store[N])
        for n = 1:N
            for k = 1:K
                grad[k, n] = 2 * real(ϕ_store[n]' * dd_store[k][n] * ψ_store[n]' * ϕ_store[N])
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
