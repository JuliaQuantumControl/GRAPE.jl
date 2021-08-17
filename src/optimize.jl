using QuantumControlBase
using QuantumPropagators
using Parameters
using ConcreteStructs

@concrete struct GrapeWrk
    objectives # copy of objectives
    pulse_mapping # as michael describes, similar to c_ops
    H_store # store for Hamiltonian
    ψ_store # store for forward states
    aux_state # store for [psi 0]
    aux_store # store for auxiliary matrix
    dP_du # store for directional derivative
end

function GrapeWrk(objective, n_slices, n_controls, T; pulse_mapping="")
    @unpack initial_state, H, target = objective
    # lets think about what we really need here
    dt = T/n_slices
    # tlist = collect(0.0:dt:T)
    dim = size(initial_state, 1)
    # store forward evolution
    ψ_store = init_storage(initial_state, n_slices+1)
    # ask Michael
    H_store = init_storage(H(0.0), n_slices)
    # aux matrix store
    aux_mat = zeros(eltype(initial_state), 2*dim, 2*dim)
    aux_state = zeros(eltype(initial_state), 2*dim)
    # storage for directional derivative, is this needed?
    dP_du = [[zeros(eltype(ψ), size(ψ)) for i = 1:N_slices] for k = 1:K_controls]

    return GrapeWrk(objective, pulse_mapping, H_store, ψ_store, aux_state, aux_mat, dP_du)
end

function optimize(wrk, pulse_options, tlist, propagator, )
    @unpack objectives, pulse_mapping, H_store, ψ_store, aux_state,aux_store, dP_du = wrk

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
