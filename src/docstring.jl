# vim: set filetype=markdown :
#
# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

@doc raw"""
Solve a quantum control problem using the GRAPE method.

```julia
using GRAPE
result = GRAPE.optimize(trajectories, tlist; J_T, kwargs...)
```

minimizes a functional

```math
J(\{ϵ_{nl}\}) = J_T(\{|Ψ_k(T)⟩\}) + λ_a J_a(\{ϵ_{nl}\})\,,
```

via the GRAPE method, where the final time functional ``J_T`` depends
explicitly on the forward-propagated states ``|Ψ_k(T)⟩``, where ``|Ψ_k(t)⟩`` is
the time evolution of the `initial_state` in the ``k``th' element of the
`trajectories`, and the running cost ``J_a`` depends explicitly on pulse values
``ϵ_{nl}`` of the l'th control discretized on the n'th interval of the time
grid `tlist`.

It does this by calculating the gradient of the final-time functional

```math
\nabla J_T \equiv \frac{\partial J_T}{\partial ϵ_{nl}}
= -2 \Re
\underbrace{%
\underbrace{\bigg\langle χ(T) \bigg\vert \hat{U}^{(k)}_{N_T} \dots \hat{U}^{(k)}_{n+1} \bigg \vert}_{\equiv \bra{\chi(t_n)}\;\text{(bw. prop.)}}
\frac{\partial \hat{U}^{(k)}_n}{\partial ϵ_{nl}}
}_{\equiv \bra{χ_k^\prime(t_{n-1})}}
\underbrace{\bigg \vert \hat{U}^{(k)}_{n-1} \dots \hat{U}^{(k)}_1 \bigg\vert Ψ_k(t=0) \bigg\rangle}_{\equiv |\Psi(t_{n-1})⟩\;\text{(fw. prop.)}}\,,
```

where ``\hat{U}^{(k)}_n`` is the time evolution operator for the ``n`` the
interval, generally assumed to be ``\hat{U}^{(k)}_n = \exp[-i \hat{H}_{kn}
dt_n]``, where ``\hat{H}_{kn}`` is the operator obtained by
evaluating `trajectories[k].generator` on the ``n``'th time interval.

The backward-propagation of ``|\chi_k(t)⟩`` has the boundary condition

```math
    |\chi_k(T)⟩ \equiv - \frac{\partial J_T}{\partial ⟨\Psi_k(T)|}\,.
```

The final-time gradient ``\nabla J_T`` is combined with the gradient for the
running costs, and the total gradient is then fed into an optimizer
(L-BFGS-B by default) that iteratively changes the values ``\{ϵ_{nl}\}`` to
minimize ``J``.

See [Background](@ref GRAPE-Background) for details.

Returns a [`GrapeResult`](@ref).

# Positional arguments

* `trajectories`: A vector of [`Trajectory`](@extref `QuantumControl.Trajectory`)
  objects. Each trajectory contains an `initial_state` and a dynamical
  `generator` (e.g., time-dependent Hamiltonian). Each
  trajectory may also contain arbitrary additional attributes like
  `target_state` to be used in the `J_T` functional
* `tlist`: A vector of time grid values.

# Required keyword arguments

* `J_T`: A function `J_T(Ψ, trajectories)` that evaluates the final time
  functional from a list `Ψ` of forward-propagated states and
  `trajectories`. The function `J_T` may also take a keyword argument
  `tau`. If it does, a vector containing the complex overlaps of the target
  states (`target_state` property of each trajectory in `trajectories`)
  with the propagated states will be passed to `J_T`.

# Optional keyword arguments

* `chi`: A function `chi(Ψ, trajectories)` that receives a list `Ψ`
  of the forward propagated states and returns a vector of states
  ``|χₖ⟩ = -∂J_T/∂⟨Ψₖ|``. If not given, it will be automatically determined
  from `J_T` via [`QuantumControl.Functionals.make_chi`](@ref) with the
  default parameters. Similarly to `J_T`, if `chi` accepts a keyword argument
  `tau`, it will be passed a vector of complex overlaps.
* `chi_min_norm=1e-100`: The minimum allowable norm for any ``|χₖ(T)⟩``.
  Smaller norms would mean that the gradient is zero, and will abort the
  optimization with an error.
* `J_a`: A function `J_a(pulsevals, tlist)` that evaluates running costs over
  the pulse values, where `pulsevals` are the vectorized values ``ϵ_{nl}``,
  where `n` are in indices of the time intervals and `l` are the indices over
  the controls, i.e., `[ϵ₁₁, ϵ₂₁, …, ϵ₁₂, ϵ₂₂, …]` (the pulse values for each
  control are contiguous). If not given, the optimization will not include a
  running cost.
* `gradient_method=:gradgen`: One of `:gradgen` (default) or `:taylor`.
  With `gradient_method=:gradgen`, the gradient is calculated using
  [QuantumGradientGenerators](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl).
  With `gradient_method=:taylor`, it is evaluated via a Taylor series, see
  Eq. (20) in Kuprov and Rogers,  J. Chem. Phys. 131, 234108
  (2009) [KuprovJCP2009](@cite).
* `taylor_grad_max_order=100`: If given with `gradient_method=:taylor`, the
  maximum number of terms in the Taylor series. If
  `taylor_grad_check_convergence=true` (default), if the Taylor series does not
  convergence within the given number of terms, throw an an error. With
  `taylor_grad_check_convergence=true`, this is the exact order of the Taylor
  series.
* `taylor_grad_tolerance=1e-16`: If given with `gradient_method=:taylor` and
  `taylor_grad_check_convergence=true`, stop the Taylor series when the norm of
  the term falls below the given tolerance. Ignored if
  `taylor_grad_check_convergence=false`.
* `taylor_grad_check_convergence=true`: If given as `true` (default), check the
  convergence after each term in the Taylor series an stop as soon as the norm
  of the term drops below the given number. If `false`, stop after exactly
  `taylor_grad_max_order` terms.
* `lambda_a=1`: A weight for the running cost `J_a`.
* `grad_J_a`: A function to calculate the gradient of `J_a`. If not given, it
  will be automatically determined. See [`make_grad_J_a`](@ref) for the
  required interface.
* `upper_bound`: An upper bound for the value of any optimized control.
  Time-dependent upper bounds can be specified via `pulse_options`.
* `lower_bound`: A lower bound for the value of any optimized control.
  Time-dependent lower bounds can be specified via `pulse_options`.
* `pulse_options`: A dictionary that maps every control (as obtained by
  [`get_controls`](@ref QuantumControl.QuantumPropagators.Controls.get_controls)
  from the `trajectories`) to a dict with the following possible keys:
  - `:upper_bounds`: A vector of upper bound values, one for each intervals of
    the time grid. Values of `Inf` indicate an unconstrained upper bound for
    that time interval, respectively the global `upper_bound`, if given.
  - `:lower_bounds`: A vector of lower bound values. Values of `-Inf` indicate
    an unconstrained lower bound for that time interval,
* `callback`: A function (or tuple of functions) that receives the
  [GRAPE workspace](@ref GrapeWrk) and the iteration number. The function
  may return a tuple of values which are stored in the
  [`GrapeResult`](@ref) object `result.records`. The function can also mutate
  the workspace, in particular the updated `pulsevals`. This may be used,
  e.g., to apply a spectral filter to the updated pulses or to perform
  similar manipulations.
* `check_convergence`: A function to check whether convergence has been reached.
  Receives a [`GrapeResult`](@ref) object `result`, and should set
  `result.converged` to `true` and `result.message` to an appropriate string in
  case of convergence. Multiple convergence checks can be performed by chaining
  functions with `∘`. The convergence check is performed after any `callback`.
* `prop_method`: The propagation method to use for each trajectory, see below.
* `verbose=false`: If `true`, print information during initialization
* `rethrow_exceptions`: By default, any exception ends the optimization, but
  still returns a [`GrapeResult`](@ref) that captures the message associated
  with the exception. This is to avoid losing results from a long-running
  optimization when an exception occurs in a later iteration. If
  `rethrow_exceptions=true`, instead of capturing the exception, it will be
  thrown normally.

# Experimental keyword arguments

The following keyword arguments may change in non-breaking releases:

* `x_tol`: Parameter for Optim.jl
* `f_tol`: Parameter for Optim.jl
* `g_tol`: Parameter for Optim.jl
* `show_trace`: Parameter for Optim.jl
* `extended_trace`:  Parameter for Optim.jl
* `show_every`: Parameter for Optim.jl
* `allow_f_increases`: Parameter for Optim.jl
* `optimizer`: An optional Optim.jl optimizer (`Optim.AbstractOptimizer`
  instance). If not given, an [L-BFGS-B](https://github.com/Gnimuc/LBFGSB.jl)
  optimizer will be used.

# Trajectory propagation

GRAPE may involve three types of time propagation, all of which are implemented via the [`QuantumPropagators`](@extref QuantumPropagators :doc:`index`) as a numerical backend:

* A forward propagation for every [`Trajectory`](@ref) in the `trajectories`
* A backward propagation for every trajectory
* A backward propagation of a
  [gradient generator](@extref QuantumGradientGenerators.GradGenerator)
  for every trajectory.

The keyword arguments for each propagation (see [`propagate`](@ref)) are
determined from any properties of each [`Trajectory`](@ref) that have a `prop_`
prefix, cf. [`init_prop_trajectory`](@ref).

In situations where different parameters are required for the forward and
backward propagation, instead of the `prop_` prefix, the `fw_prop_` and
`bw_prop_` prefix can be used, respectively. These override any setting with
the `prop_` prefix. Similarly, properties for the backward propagation of the
gradient generators can be set with properties that have a `grad_prop_` prefix.
These prefixes apply both to the properties of each [`Trajectory`](@ref) and
the keyword arguments.

Note that the propagation method for each propagation must be specified. In
most cases, it is sufficient (and recommended) to pass a global `prop_method`
keyword argument.
""" optimize
