using QuantumControlBase
using QuantumPropagators
using Parameters
using ConcreteStructs
using Optim

import LinearAlgebra


@doc raw"""Extended generator for the standard dynamic gradient.

```julia
G̃ = GradGenerator(G)
```

contains the original generator `G` (a Hamiltonian or Liouvillian) in `G̃.G`, a
vector of control generators ``∂G/∂ϵᵢ`` (static operators for linear controls,
and functions for non-linear controls) in `G̃.control_generators`, and the
controls in `G̃.controls`.

For a generator ``G = Ĥ = Ĥ₀ + ϵ₁(t) Ĥ₁ + … +  ϵₙ(t) Ĥₙ``, this extended
generator encodes the block-matrix

```math
G̃ = \begin{pmatrix}
         Ĥ  &  0 &  \dots   &  0  &  Ĥ₁     \\
         0  &  Ĥ &  \dots   &  0  &  Ĥ₂     \\
    \vdots  &    &  \ddots  &     &  \vdots \\
         0  &  0 &  \dots   &  Ĥ  &  Ĥₙ     \\
         0  &  0 &  \dots   &  0  &  Ĥ
\end{pmatrix}
```
"""
struct GradGenerator{T, CGT, CT}
    G :: T
    control_generators :: Vector{CGT}
    controls :: Vector{CT}

    function GradGenerator(G::T) where T
        control_generators = G -> [] # TODO implement alongside getcontrols
        G_ctrl = control_generators(G)
        controls = collect(getcontrols(G))
        new{T, eltype(G_ctrl), eltype(controls)}(G, G_ctrl, controls)
    end

end


@doc raw"""Extended state-vector for the dynamic gradient.

```julia
Ψ̃ = GradVector(Ψ, G̃)
```

for a state `Ψ` and a [`GradGenerator`](@ref) `G̃` contains the original `Ψ` in
`Ψ̃.state` and storage for ``n`` gradient-states for the `n` controls in `G̃`
in `Ψ̃.grad_states`.

The `GradVector` conceptually corresponds to a direct-sum (block) column-vector
``Ψ̃ = (|Ψ̃₁⟩, |Ψ̃₂⟩, … |Ψ̃ₙ⟩, |Ψ⟩)^T``. With a ``G̃`` as in the documentation of
[`GradGenerator`](@ref), we have

```math
G̃ Ψ̃ = \begin{pmatrix}
Ĥ |Ψ̃₁⟩ + Ĥ₁|Ψ⟩ \\
\vdots \\
Ĥ |Ψ̃ₙ⟩ + Ĥₙ|Ψ⟩ \\
Ĥ |Ψ⟩
\end{pmatrix}
```

and

```math
e^{-i G̃ dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
\frac{∂}{∂ϵ₁} e^{-i Ĥ dt} |Ψ⟩ \\
\vdots \\
\frac{∂}{∂ϵₙ} e^{-i Ĥ dt} |Ψ⟩ \\
e^{-i Ĥ dt} |Ψ⟩
\end{pmatrix}.
```
"""
struct GradVector{T}
    state :: T
    grad_states :: Vector{T}
end

function GradVector(Ψ::T, G::GradGenerator) where T
    grad_states = [similar(Ψ) for _ in 1:length(G.control_generators)]
    for i = 1 : length(grad_states)
        LinearAlgebra.lmul!(0.0, grad_states[i])
    end
    GradVector{T}(Ψ, grad_states)
end


function LinearAlgebra.mul!(Φ::GradVector, G::GradGenerator, Ψ::GradVector)
    LinearAlgebra.mul!(Φ.state, G.G, Ψ.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[i], G.G, Ψ.grad_states[i])
        LinearAlgebra.mul!(Φ.grad_states[i], G.control_generators[i], Ψ.state, 1, 1)
    end
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[i])
    end
end

function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy(a, X.state, Y.state)
    for i = 1:length(X.grad_states)
        LinearAlgebra.axpy(a, X.grad_states[i], Y.grad_states[i])
    end
end


function LinearAlgebra.norm(Ψ::GradVector)
    nrm = LinearAlgebra.norm(Ψ.state)
    for i = 1:length(Ψ.grad_states)
        nrm += LinearAlgebra.norm(Ψ.grad_states[i])
    end
    return nrm
end


function Base.copyto!(dest::GradVector, src::GradVector)
    copyto!(dest.state, src.state)
    for i = 1:length(src.grad_states)
        copyto!(dest.grad_states[i], src.grad_states[i])
    end
    return dest
end


function Base.length(Ψ::GradVector)
    return length(Ψ.state) * (1 + length(Ψ.grad_states))
end


function Base.similar(Ψ::GradVector)
    return GradVector(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
end


@concrete struct GrapeWrk
    objectives # copy of objectives
    pulse_mapping # as michael describes, similar to c_ops
    H_store # store for Hamiltonian
    ψ_store # store for forward states
    ϕ_store # store for forward states
    # aux_state # store for [psi 0]
    # aux_store # store for auxiliary matrix
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
    # store for forward evolution
    ψ_store = [[similar(initial_state) for i = 1:N_slices] for ii = 1:N_obj]

    # set the state in each first entry to be the initial state
    for i = 1:N_obj
        ψ_store[i][1] = objectives[i].initial_state
    end

    # store for the backward evolution
    ϕ_store = [[similar(initial_state) for i = 1:N_slices] for ii = 1:N_obj]

    # similarly we set the final entry of each to the target state
    for i = 1:N_obj
        ϕ_store[i][N_slices] = objectives[i].target_state
    end

    # storage for the Hamiltonian
    H_store = [[similar(generator[1]) for i = 1:N_slices] for ii = 1:N_obj]

    # store for the directional derivative
    dP_du = [
        [[zeros(eltype(ψ), size(ψ)) for i = 1:N_slices] for k = 1:length(controls)] for
        ii = 1:N_obj
    ]

    # propagator working structs, many for each objective
    prop_wrk = [initpropwrk(obj.initial_state, tlist; prop_method) for obj in objectives]

    # similar propagator working structs but for the auxilliary matrix
    aux_state_dummy = similar([objectives[1].initial_state; objectives[1].initial_state])
    aux_prop_wrk = [initpropwrk(aux_state_dummy, tlist; prop_method) for obj in objectives]
    return GrapeWrk(
        objectives,
        pulse_mapping,
        H_store,
        ψ_store,
        ϕ_store,
        dP_du,
        tlist,
        prop_wrk,
        aux_prop_wrk,
    )
end

function optimize(wrk, pulse_options)
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
        # we can do this in parallel
        @inbounds for obj = 1:N_obj
            # forward prop all states for current objective
            ψ_store_copy = copy(ψ_store[obj])
            _fw_prop!(x, ψ_store[obj], H_store[obj], N_slices, dt, prop_wrk[obj], H)
            # forward propagate the aux matrix
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
            # backward propagate the costate and store it at each point in time
            _bw_prop!(ϕ_store[obj], H_store[obj], N_slices, dt, prop_wrk[obj])
            # compute the fidelity 
            fid = abs2(ϕ_store[obj][N_slices]' * ψ_store[obj][N_slices])
            
            # then compute the gradient
            @inbounds for n = 1:N_slices
                @inbounds for k = 1:N_controls
                    grad[k][n] = 2 * real(ϕ_store[obj][n]' * dP_du[obj][k][n])
                end
            end

            # add scaled gradient
            if G !== nothing
                G .= +grad / N_obj
            end
            # sum figure of merit, needs weights
            if F !== nothing
                F = F + fid / N_obj
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
    # we minimize the result and return
    minimize(Optim.onlyfg!(topt), pulses, LBFGS())


end


"""
Propagate system forward in time and store states at a constant objective index
"""
function _fw_prop!(x, ψ_store, H_store, N_slices, dt, prop_wrk, H)
    @inbounds for n = 1:N_slices
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
    @inbounds for n = 1:N_slices
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
