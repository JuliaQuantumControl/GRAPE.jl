
# basically a scratch file for now as I figure out how I want this to work 

# lets assume that first of all we want to start with simple population transfer

using LinearAlgebra
# initial state
ρ0 = [1 0;0 0] .+ 0.0im
ρT = [0 0;0 1] .+ 0.0im

# we have only sx drive
const sx = [0 1;1 0] .+ 0.0im
const sz = [1 0;0 -1] .+ 0.0im

T = 1.0
dt = 0.1
N = floor(Int, T/dt)
k = 1 # number of controls

drive = rand(k, N)

# we should remember that our Hamiltonian is some callable type
# we need to convert time to index inside right now
H(drive, t, T, N) = drive[floor(Int, t/T * N)] * sx

# H(t) = H0 + iR  + ∑ₖ c_k(t) H_k
# N slices in time
# so control vector is rand(k, N)
# reshapes control vector into 
# can then do something similar to Goodwin's thesis and use density matrix vectorisation

c_vec = reshape(drive, (k * N, 1))

# propagator 
# Phat_n (c_n) = exp( - 1.0im * (H0 + iR + ∑_k c_{k, n} H_k) Δt)
# where c_n is the matrix representing the coefficients of each control at time 

# H_n is the full Ham at slice n
# H_k is the control Ham correspondning to k

z = [ρ0]
for n = 1:N
    ρ = z[n]
    H_n = H(drive .* 0.0 .+ 0.5 * pi, n * dt, T, N)
    H_k = sx
    blz = zero(H_k)
    
    L2 = [H_n H_k; blz H_n]
    expl2 = exp(-1.0im * L2 * dt)

    DH_k = expl2[1:2, 3:end]
    P_n = expl2[1:2, 1:2]

    ρ_t = P_n * ρ * P_n'
    append!(z, [ρ_t])

end


zz = [real(tr(i * sz)) for i in z]
@show zz

# H_n = H(drive, 0.1, T, N)
# H_k = sx
# blz = zero(H_k)


# L2 = [H_n H_k; blz H_n]

# expl2 = exp(-1.0im * L2 * dt)

# DH_k = expl2[1:2, 3:end]
# P_n = expl2[1:2, 1:2]



# work in vector formalism

ρ0vec = reshape(ρ0, (4,1))
ρTvec = reshape(ρT, (4, 1))

H_n = H(drive, 0.1, T, N)
i2 = I(size(H_n,1))

Hhat = kron(i2, H_n) - kron(H_n', i2)


z = [ρ0vec]

for n = 1:N
    ρ = z[n]
    H_n_temp = H(drive .* 0.0 .+ 0.5 * pi, n * dt, T, N)
    H_n = kron(i2, H_n_temp) - kron(H_n_temp', i2)

    i2 = I(size(H_n_temp,1))

    H_k = kron(i2, sx) - kron(sx', i2)

    
    blz = zero(H_k)
    
    L2 = [H_n H_k; blz H_n]
    expl2 = exp(-1.0im * L2 * dt)

    # DH_k = expl2[1:2, 3:end]
    P_n = expl2[1:4, 1:4]

    ρ_t = P_n * ρ
    append!(z, [ρ_t])

end


szhat = kron(i2, sz) - kron(sz', i2)

zz = [real(tr(sz * reshape(i, (2,2)))) for i in z]

@show zz

# from discussion with Michael

# from notes from goodwin thesis (spinach)
# do forward prop with arnoldi to compute states going forward
# do backward prop with arnoldi to compute costates
# compute fidelity
# for gradient
# reuse forward prop states
# aux matrix to get rhs of derivative
# then use costates to get gradient value


# okay so using Arnoldi from ExpUtils for now
using ExponentialUtilities








ρ0vec = reshape(ρ0, (4))
ρTvec = reshape(ρT, (4))

H_n = H(drive, 0.1, T, N)
i2 = I(size(H_n,1))

Hhat = kron(i2, H_n) - kron(H_n', i2)

# ahead of time we allocate storage arrays for states
state = [ρ0vec for i=1:N+1]
costate = [ρTvec for i = 1:N+1]
# storage array for the directional derivative
blz = zero(ρ0vec)

# also need to remember the kth index for the states
directderiv_store = [ρ0vec for i = 1:N+1]

for n = 1:N
    ρ = state[n]
    H_n_temp = H(drive .* 0.0 .+ 0.5 * pi, n * dt, T, N)
    i2 = I(size(H_n_temp,1))
    H_n = kron(i2, H_n_temp) - kron(H_n_temp', i2)

    H_k = kron(i2, sx) - kron(sx', i2)    
    blz = zero(H_k)

    ρ_t = expv(-1.0im * dt, H_n, ρ)
    state[n+1] = ρ_t

    L2 = [H_n H_k; blz H_n]
    fakestate = [zero(ρ); ρ]

    test = expv(-1.0im * dt, L2, fakestate)
    directderiv_store[n] = test[1:4]
    # append!(z, [ρ_t])
    
    # expl2 = exp(-1.0im * L2 * dt)

    # # DH_k = expl2[1:2, 3:end]
    # P_n = expl2[1:4, 1:4]

    # ρ_t = P_n * ρ
end

for n = reverse(1:N)
    χ = costate[n + 1]

    H_n_temp = H(drive .* 0.0 .+ 0.5 * pi, n * dt, T, N)
    i2 = I(size(H_n_temp,1))
    H_n = kron(i2, H_n_temp) - kron(H_n_temp', i2)

    H_k = kron(i2, sx) - kron(sx', i2)    
    blz = zero(H_k)

    χ_t = expv(1.0im * dt, H_n, χ)
    costate[n] = χ_t
end

# then finally the gradient is computed quite simply using the costates and the directional derivatives




szhat = kron(i2, sz) - kron(sz', i2)

zz = [real(tr(sz * reshape(i, (2,2)))) for i in z]

zzz = [real(tr(sz * reshape(i, (2,2)))) for i in costate]



struct SchirmerOp
    H
    μ
end

struct SchirmerState
    ψ
    ϕ
end

function mul!(ψ_out::SchirmerState, H::SchirmerOp, ψ::SchirmerState)

end