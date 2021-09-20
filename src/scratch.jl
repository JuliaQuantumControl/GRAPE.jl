
# basically a scratch file for now as I figure out how I want this to work 

# lets assume that first of all we want to start with simple population transfer
using QuantumPropagators
using LinearAlgebra
using QuantumControlBase
using Parameters


# most of this is a terrible idea


"""Two-level-system Hamiltonian."""
function hamiltonian(drive_array, Ω=1.0)
    σ̂_z = ComplexF64[1 0; 0 -1];
    σ̂_x = ComplexF64[0 1; 1  0];
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

tlist = collect(range(0, T, length=n_slices))
objectives = [Objective(initial_state = ρ0, generator = H, target = ρT)]


# propagate final state backwards
# propagate initial state forwards step by step
# compute the result
# compute the gradient

problem = ControlProblem(objectives = objectives, pulse_options = nothing, tlist=tlist, prop_method = :expprop)
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
pulses = [
    discretize_on_midpoints(control, tlist) for control in controls
]
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
        wrk.aux_store[obj][1:dim, dim+1:end] .= 
    end
    # then we need to take a propstep
end

# so we have the forward evolved states, now we backwards evolve our final state
# and we also evolve our auxilliary state using the auxilliary matrix


# then at some point we have a hot loop where all this happens



# we have only sx drive
const sx = [0 1;1 0] .+ 0.0im
const sz = [1 0;0 -1] .+ 0.0im
const sy = [0 -1.0im; 1.0im 0]
const eye2 = I(2)

super_sx = kron(eye2, sx) - kron(sx', eye2)
super_sy = kron(eye2, sy) - kron(sy', eye2)

T = 1.0
dt = 0.1
N = floor(Int, T/dt)
K = 2 # number of controls

drive = rand(k, N)
grad = zero(drive)

# we should remember that our Hamiltonian is some callable type
# we need to convert time to index inside right now
H(drive, t, T, N) = drive[1, floor(Int, t/T * N),] * sx + drive[2, floor(Int, t/T * N),] * sy

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
    return dim, ψ_store, ϕ_store, zerofs, temp_state, zeromat, aux_mat, directderiv_store
end


ρ0vec = reshape(ρ0, (4))
ρTvec = reshape(ρT, (4))


dim, ψ_store, ϕ_store, state0, temp_state, mat0, aux_store, dd_store = _get_storage_arrays(ρ0vec, ρTvec, N, K)

function test_grape(F, G, x, dim, ψ_store, ϕ_store, state0, temp_state, mat0, aux_store, dd_store, grad, H_func, H_K_super)
    for n = 1:N
        ψ = ψ_store[n]
        # could compute Hamiltonian once and store it
        H_n_temp = H_func(x, n * dt, T, N)
        id = I(size(H_n_temp,1))
        
        H_n = kron(id, H_n_temp) - kron(H_n_temp', id)
        # U * ψ
        ψ_store[n+1] .= expv(-1.0im * dt, H_n, ψ)
    
        # compute and store the directional derivative
        # set up the temp_state 
        temp_state[dim+1:end] .= ψ
        for k = 1:K
            # set up augmented matrix
            H_k = H_K_super[k]
            aux_store[1:dim, dim+1:end] .= H_k
            # then on the diagonal we want H_n
            aux_store[1:dim, 1:dim] .= H_n
            aux_store[dim+1:end, dim+1:end] .= H_n
            dd_store[k][n] .= expv(-1.0im * dt, aux_store, temp_state)[1:dim]
        end
    end

    @show abs2.(ψ_store[end])

    for n = reverse(1:N)
        ϕ = ϕ_store[n+1]
        H_n_temp = H_func(x, n * dt, T, N)
        id = I(size(H_n_temp,1))
        
        H_n = kron(id, H_n_temp) - kron(H_n_temp', id)
        ϕ_store[n] .= expv(1.0im * dt, H_n, ϕ)
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

using Optim
topt = (F, G, x) -> test_grape(F, G, x, dim, ψ_store, ϕ_store, state0, temp_state, mat0, aux_store, dd_store, grad, H, H_K_super)


topt(1.0, nothing, res.minimizer)

res = Optim.optimize(Optim.only_fg!(topt), drive, Optim.LBFGS(), Optim.Options(show_trace=true, f_tol = 1e-3))


res.minimizer



H_K_super = [super_sx, super_sy]
# forward evolution and directional derivative


# now backwards

# compute the fidelity
fid = abs2(ϕ_store[N]' * ψ_store[N])
for n = 1:N
    for k = 1:K
        grad[k, n] = 2 * real(ϕ_store[n]' * dd_store[k][n] * ψ_store[n]' * ϕ_store[N])
    end
end

@show "hi"

# # ahead of time we allocate storage arrays for states
# state = [ρ0vec for i=1:N+1]
# costate = [ρTvec for i = 1:N+1]
# # storage array for the directional derivative
# blz = zero(ρ0vec)

# # also need to remember the kth index for the states
# directderiv_store = [ρ0vec for i = 1:N+1]

# struct SchirmerOp
#     H
#     μ
# end

# struct SchirmerState
#     ψ
#     ϕ
# end

# function mul!(ψ_out::SchirmerState, H::SchirmerOp, ψ::SchirmerState)

# end
zz = [real(tr(sz * reshape(i, (2,2)))) for i in z]






using QuantumPropagators

# lets pretend we want to evolve a state forward under some Hamiltonian

using LinearAlgebra
# initial state
ρ0 = [1 0;0 0] .+ 0.0im
ρT = [0 0;0 1] .+ 0.0im

ρ0vec = reshape(ρ0, (4))

# we have only sx drive
const sx = [0 1;1 0] .+ 0.0im
const sz = [1 0;0 -1] .+ 0.0im
const sy = [0 -1.0im; 1.0im 0]
const eye2 = I(2)

T = 1.0
dt = 0.1
N = floor(Int, T/dt)
K = 2 # number of controls

drive = rand(K, N) .* 0.0 .+ 0.5 * pi

# we should remember that our Hamiltonian is some callable type
# we need to convert time to index inside right now
super_sx = kron(eye2, sx) - kron(sx', eye2)
super_sy = kron(eye2, sy) - kron(sy', eye2)


function get_genfunc(drive, sx, sy)
    gen_func = (tlist, i; kwargs...) -> (drive[1, i] * sx )
end

gen_func = get_genfunc(drive, super_sx, super_sy)

tlist = collect(0:dt:T)


wrk = initpropwrk(ρ0vec, tlist, nothing; method=:newton)

test = propagate(ρ0vec, gen_func, tlist; wrk = wrk)


using BenchmarkTools

@benchmark propagate($ρ0vec, $gen_func, $tlist; wrk = $wrk)


using ExponentialUtilities


function test_prop(state, gen_func, tlist)
    ev = copy(state)

    for i = 1:length(tlist)
        H = gen_func(tlist, i)
        ev .= expv(-1.0im * dt, H, ev)
    end
    ev
end


test2 = test_prop(ρ0vec, gen_func, tlist)


@benchmark test_prop($ρ0vec, $gen_func, $tlist)
