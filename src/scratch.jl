
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
    H_n = H(drive, n * dt, T, N)
    H_k = sx
    blz = zero(H_k)
    
    L2 = [H_n H_k; blz H_n]
    expl2 = exp(-1.0im * L2 * dt)

    DH_k = expl2[1:2, 3:end]
    P_n = expl2[1:2, 1:2]

    ρ_t = P_n * ρ * P_n'
    append!(z, [ρ_t])

end


# zz = [real(tr(i * sz)) for i in z]

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