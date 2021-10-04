
# basically a scratch file for now as I figure out how I want this to work 

# lets assume that first of all we want to start with simple population transfer
using QuantumPropagators
using LinearAlgebra
using QuantumControlBase
using Parameters


# most of this is a terrible idea


"""Two-level-system Hamiltonian."""
function hamiltonian(drive_array, Ω = 1.0)
    σ̂_z = ComplexF64[1 0; 0 -1]
    σ̂_x = ComplexF64[0 1; 1 0]
    Ĥ₀ = -0.5 * Ω * σ̂_z
    Ĥ₁ = σ̂_x
    return (Ĥ₀, (Ĥ₁, drive_array))
end

# initial state
ρ0 = [1, 0] .+ 0.0im
ρT = [0, 1] .+ 0.0im


n_slices = 10
T = 5.0

drive = rand(n_slices)

H = hamiltonian(drive)

tlist = collect(range(0, T, length = n_slices))
objectives = [Objective(initial_state = ρ0, generator = H, target_state = ρT)]


problem = ControlProblem(
    objectives = objectives,
    pulse_options = Dict(),
    tlist = tlist,
    prop_method = :newton,
)

wrk = GrapeWrk(objectives, tlist, :newton)

# pulses must be passed as arg
controls = getcontrols(objectives)
pulses = [discretize_on_midpoints(control, tlist) for control in controls]
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


grad = [similar(pulses[1])]

N_slices = length(tlist) - 1
dim = size(H_store[1][1], 1)
N_controls = size(controls, 1)
dt = tlist[2] - tlist[1]


N_obj = length(objectives)
obj = 1

@inbounds for n = 1:N_slices
    # save the initial state in each timestep
    ψ_not_mutated = copy(ψ_store[obj][n])
    # copy the state into the next slice since propstep! will mutate it
    ψ_store[obj][n+1] .= ψ_store[obj][n]
    # save the Hamiltonian for computation later
    H_store[obj][n] .= H[1] + H[2][1] .* pulses[1][n]
    ψ = ψ_store[obj][n+1]
    propstep!(ψ, H_store[obj][n], dt, prop_wrk[obj])

    aux_state[obj][dim+1:end] .= ψ_not_mutated
    # dP_du[obj][k][n] .= ψ_not_mutated
    @inbounds for k = 1:N_controls
        aux_store[obj][1:dim, dim+1:end] .= H[2][1]
        aux_store[obj][1:dim, 1:dim] .= H_store[obj][n]
        aux_store[obj][dim+1:end, dim+1:end] .= H_store[obj][n]
        propstep!(aux_state[obj], aux_store[obj], dt, aux_prop_wrk[obj])
        dP_du[obj][k][n] .= aux_state[obj][1:dim]
        aux_state[obj] .= 0.0 + 0.0im
    end
end

@inbounds for n in reverse(1:N_slices)
    # copy your current state into this slot so it can be mutated
    ϕ_store[obj][n] .= ϕ_store[obj][n+1]
    ϕ = ϕ_store[obj][n]
    # and then prop and mutate
    # we pass a negative dt in the hope we can move it the opposite direction in time
    propstep!(ϕ, H_store[obj][n], -1.0 * dt, prop_wrk[obj])
end

fid = abs2(ϕ_store[obj][N_slices]' * ψ_store[obj][N_slices])

n = N_slices
k = N_controls
2 * real(ϕ_store[obj][n]' * dP_du[obj][k][n])


for n = 1:N_slices
    for k = 1:N_controls
        grad[k][n] = 2 * real(ϕ_store[obj][n]' * dP_du[obj][k][n])
    end
end

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
    H_func,
    H_K_super,
)
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

    for n in reverse(1:N)
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







# propagate final state backwards
# propagate initial state forwards step by step
# compute the result
# compute the gradient

problem = ControlProblem(
    objectives = objectives,
    pulse_options = nothing,
    tlist = tlist,
    prop_method = :newton,
)
@unpack objectives, pulse_options, tlist = problem
prop_method = get(problem.kwargs, :prop_method, :auto)
# then we create the storage arrays and things that we need
wrk = GrapeWrk3(objectives, tlist)

# now we set up a function to optimize
N_obj = length(objectives)
obj = 1
# for obj = 1:N_obj
# end
controls = getcontrols(objectives)
pulses = [discretize_on_midpoints(control, tlist) for control in controls]
# obviously need to convert this to Michaels if I ever understand it
ψ = objectives[obj].initial_state
dt = tlist[2] - tlist[1]

dim = 2
# now loop over time
for i = 1:size(pulses[1])[1]
    # because the propstep! method replaces ψ_store
    wrk.ψ_store[obj][i+1] .= wrk.ψ_store[obj][i]

    # we store the aux_state
    wrk.aux_state[obj][dim+1:end] .= ψ
    # we end up in this horrific situation
    gen = H[1] + H[2][1] .* pulses[1][i]
    ψ = wrk.ψ_store[obj][i+1]
    propstep!(ψ, gen, dt, wrk.prop_wrk[obj])
    # and lets compute the forward evolution of the aux state

    for k = 1:2
        wrk.aux_store[obj][1:dim, dim+1:end] .= 1.0
    end
    # then we need to take a propstep
end

# so we have the forward evolved states, now we backwards evolve our final state
# and we also evolve our auxilliary state using the auxilliary matrix


# then at some point we have a hot loop where all this happens

