# Background

```@contents
Pages=["background.md"]
Depth=2:3
```

## Introduction

The GRAPE methods minimizes an optimization functional of the form

```math
\begin{equation}\label{eq:grape-functional}
J(\{ϵ_l(t)\})
    = J_T(\{|Ψ_k(T)⟩\})
    + λ_a \, \underbrace{∑_l \int_{0}^{T} g_a(ϵ_l(t)) \, dt}_{=J_a(\{ϵ_l(t)\})}
    + λ_b \, \underbrace{∑_k \int_{0}^{T} g_b(|Ψ_k(t)⟩) \, dt}_{=J_b(\{|Ψ_k(t)⟩\})}\,,
\end{equation}
```

where ``\{ϵ_l(t)\}`` is a set of [control functions](@extref QuantumControl :label:`Control-Function`) defined between the initial time ``t=0`` and the final time ``t=T``, and ``\{|Ψ_k(t)⟩\}`` is a set of ["trajectories"](@extref QuantumControl.Trajectory) evolving from a set of initial states ``\{|\Psi_k(t=0)⟩\}`` under the controls ``\{ϵ_l(t)\}``. The primary focus is on the final-time functional ``J_T``, but running costs ``J_a`` (weighted by ``λ_a``) may be included to penalize certain features of the control field. In principle, a state-dependent running cost ``J_b`` weighted by ``λ_b`` can also be included (and will be discussed below), although this is currently not fully implemented in `GRAPE.jl`.

The defining assumptions of the GRAPE method are

1.  The control fields ``\epsilon_l(t)`` are piecewise-constant on the ``N_T`` intervals of a time grid ``t_0 = 0, t_1, \dots t_{N_T} = T``. That is, we have a vector of pulse values with elements ``\epsilon_{nl}``. We use the double-index `nl`, for the value of the ``l``'th control field on the ``n``'th interval of the time grid.

2. The states ``\ket{\Psi_k(t)}`` evolve under an equation of motion of the form

```math
\begin{equation}\label{eq:tdse}
    i \hbar \frac{\partial \ket{\Psi_k(t)}}{\partial t} = \hat{H}_k(\{\epsilon_l(t)\}) \ket{\Psi_k(t)}\,,
\end{equation}
```

with ``\hbar = 1``.

This includes the Schrödinger equation, but also the Liouville equation for open quantum systems. In the latter case ``\ket{\Psi_k}`` is replaced by a vectorized density matrix, and ``\hat{H}_k`` is replaced by a Liouvillian (super-) operator describing the dynamics of the ``k``'th trajectory. The crucial point is that Eq. \eqref{eq:tdse} can be solved analytically within each time interval as

```math
\begin{equation}\label{eq:time-evolution-op}
    \def\ii{\mathrm{i}}
    \ket{\Psi_k(t_{n+1})} = \underbrace{\exp\left[-\ii \hat{H}_{kn} dt_n \right]}_{=\hat{U}^{(k)}_{n}} \ket{\Psi_k(t_n)}\,,
\end{equation}
```

where ``\hat{H}_{kn} = \hat{H}_k(\{\epsilon_{nl}\})`` is ``\hat{H}_k(\{\epsilon_l(t)\})`` evaluated at the midpoint of the ``n``'th interval (respectively at ``t=0`` and ``t=T`` for ``n=1`` and ``n=N_T``), and with the time step ``dt_n = (t_n - t_{n-1})``.


These two assumptions allow to analytically derive the gradient ``(\nabla J)_{nl} \equiv \frac{\partial J}{\partial \epsilon_{nl}}``. The initial derivation of GRAPE by [KhanejaJMR2005](@citet) focuses on a final-time functional ``J_T`` that depends of the overlap of each forward-propagated ``|\Psi_k(T)⟩`` with a target state ``|\Psi^{\text{tgt}}_k(T)⟩`` and updates the pulse values ``\epsilon_{nl}`` directly in the direction of the negative gradient. Improving on this, [FouquieresJMR2011](@citet) showed that using a quasi-Newton method to update the pulses based on the gradient information leads to a dramatic improvement in convergence and stability. Furthermore, [GoodwinJCP2015](@citet) improved on the precision of evaluating the gradient of a local time evolution operator, which is a critical step in the GRAPE scheme. Finally, [GoerzQ2022](@citet) generalized GRAPE to arbitrary functionals of the form \eqref{eq:grape-functional}, bridging the gap to automatic differentiation techniques [LeungPRA2017, AbdelhafezPRA2019, AbdelhafezPRA2020](@cite) by introducing the technique of "semi-automatic differentiation". This most general derivation is the basis for the implementation in `GRAPE.jl`.

```@raw html
<a id="tmidr"/>
```

!!! tip "Too Many Indices; Didn't Read (TMIDR)"

    Below, we derive the GRAPE scheme here in full generality. This implies keeping track of a lot of indices:

    * ``k``: the index over the different [trajectories](@extref QuantumControl.Trajectory), i.e. the states ``|\Psi_k(t)⟩`` whose time evolution contribute to the functional
    * ``l``: the index over the different [control functions](@extref QuantumControl label:`Control-Function`) ``\epsilon_l(t)`` that the Hamiltonian/Liouvillian may depend on
    * ``n``: The index over the intervals of the time grid

    Most equations can be simplified by not worrying about ``k`` or ``l``: If there multiple controls, they are concatenated into a single vector of control values with a double-index ``nl``. We really only need to keep track of ``n``; the gradient values related to a ``\epsilon_{nl}`` with a particular ``l`` are somewhat obvioulsly obtained by using a particular ``\epsilon_l(t_n)``. Likewise, all trajectories contribute equally to the gradients, so we just have a sum over the ``k`` index.

    We can further simplify by considering only final-time functionals ``J_T(\{|\Psi_k(T)⟩\})``. Running costs ``J_a(\{ϵ_l(t)\})`` are quite straightforward to add (just take the deriverive w.r.t. the values ``ϵ_{nl}``), and running costs ``J_b(\{|Ψ_k(t)⟩\})`` are too complicated to consider in any kind of "simplified" scheme.

    In essence, then, the GRAPE scheme that is implemented here can then be concisely summarized, cf. Eq. \eqref{eq:grad-at-T-U}, as

    ```math
    \begin{equation}
    \frac{\partial J_T}{\partial \epsilon_{n}}
    = -2 \Re
       \underbrace{%
           \underbrace{\bigg\langle \chi(T) \bigg\vert \hat{U}_{N_T} \dots \hat{U}_{n+1} \bigg \vert}_{\equiv \bra{\chi(t_n)}\;\text{(bw. prop.)}}
           \frac{\partial \hat{U}_n}{\partial \epsilon_{n}}
       }_{\equiv \bra{\chi^\prime(t_{n-1})}}
       \underbrace{\bigg \vert \hat{U}_{n-1} \dots \hat{U}_1 \bigg\vert \Psi(t=0) \bigg\rangle}_{\equiv \ket{\Psi(t_{n-1})}\;\text{(fw. prop.)}}\,,
    \end{equation}
    ```

    with the boundary condition, cf. Eq. \eqref{eq:chi},

    ```math
    \begin{equation}
        |\chi(T)⟩ \equiv - \frac{\partial J_T}{\partial ⟨\Psi(T)|}\,.
    \end{equation}
    ```

    The gradient-state ``|\chi^\prime(t_{n-1})⟩`` is obtained either via an expansion of ``\hat{U}_n`` [into a Taylor series](@ref Overview-Taylor), or (by default), by backward-propagating an [extended state ``|\tilde\chi(t)⟩`` with gradient information](@ref Overview-Gradgen) [GoodwinJCP2015](@cite). The resulting scheme is illustrated in [Fig. 1](#fig-grape-scheme).


## Prerequisite: Wirtinger derivatives and matrix calculus

Even though we are seeking the derivative of the real-valued functional ``J`` with respect to the real-valued parameter ``\epsilon_{nl}``, the functional still involves complex quantities via ``|\Psi_k(t)⟩`` and ``\hat{H}`` in Eq. \eqref{eq:tdse}. In order to apply the chain rule in the derivation of the gradient, we will have to clarify the notion of derivatives in the context of complex numbers, as well as derivatives with respect to vectors ("matrix calculus").

### Derivatives w.r.t. complex scalars

To illustrate, let's say we introduce intermediary scalar variables ``z_k \in \mathbb{C}`` in the functional, ``J(\{\epsilon_{nl}\}) \rightarrow J(\{z_k(\{\epsilon_{nl}\})\})``, with ``J, \epsilon_{nl} \in \mathbb{R}``.

In principle, one must separate the ``z_k`` into real and imaginary part as independent variables, ``J = J(\{\Re[z_k]\}, \{\Im[z_k]\})``, resulting in

```math
\begin{equation}
  \label{eq:grad_zj_real_imag}
  (\nabla J)_{nl}
  \equiv \frac{\partial J}{\partial \epsilon_{nl}}
  = \sum_k \left(
    \frac{\partial J}{\partial \Re[z_k]}
    \frac{\partial \Re[z_k]}{\partial \epsilon_{nl}}
    + \frac{\partial J}{\partial \Im[z_k]}
    \frac{\partial \Im[z_k]}{\partial \epsilon_{nl}}
    \right)\,.
\end{equation}
```

An elegant alternative is to introduce [Wirtinger derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives),

```math
\begin{align}%
  \label{eq:wirtinger1}
  \frac{\partial J}{\partial z_k}
    &\equiv \frac{1}{2} \left(
      \frac{\partial J}{\partial \Re[z_k]}
      -\ii \frac{\partial J}{\partial \Im[z_k]}
      \right)\,, \\
  \label{eq:wirtinger2}
  \frac{\partial J}{\partial z_k^*}
    &\equiv \frac{1}{2} \left(
      \frac{\partial J}{\partial \Re[z_k]}
      +\ii \frac{\partial J}{\partial \Im[z_k]}
      \right)
    = \left(\frac{\partial J}{\partial z_k}\right)^*\,,
\end{align}
```

which instead treats ``z_k`` and the conjugate value ``z_k^*`` as independent variables, so that

```math
\begin{equation}%
  \label{eq:wirtinger_chainrule}
  \frac{\partial J}{\partial \epsilon_{nl}}
  = \sum_k \left(
    \frac{\partial J}{\partial z_k}
    \frac{\partial z_k}{\partial \epsilon_{nl}}
    + \frac{\partial J}{\partial z_k^*}
    \frac{\partial z_k^*}{\partial \epsilon_{nl}}
    \right)
  = 2 \Re \sum_k \frac{\partial J}{\partial z_k}
    \frac{\partial z_k}{\partial \epsilon_{nl}}
    \,.
\end{equation}
```

So, we have a simple chain rules, modified only by ``2 \Re[…]``, where we can otherwise "forget" that ``z_k`` is a complex variable. The fact that ``J \in \mathbb{R}`` guarantees that ``z_k`` and ``z_k^*`` can only occur in such ways that we don't have to worry about having "lost" ``z_k^*``.

The derivative of the complex value ``z_k`` with respect to the real value ``\epsilon_{nl}`` is defined straightforwardly as

```math
\begin{equation}
  \frac{\partial z_k}{\partial \epsilon_{nl}}
  \equiv
  \frac{\partial \Re[z_k]}{\partial \epsilon_{nl}}
  + \ii \frac{\partial \Im[z_k]}{\partial \epsilon_{nl}}\,.
\end{equation}
```

### Derivatives w.r.t. complex vectors

We can now go one step further and allow for intermediate variables that are complex _vectors_ instead of scalars, ``J(\{\epsilon_{nl}\}) \rightarrow J(\{|\Psi_k(\{\epsilon_{nl}\})⟩\})``. Taking the derivative w.r.t. a vector puts us in the domain of [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus). Fundamentally, the derivative of a scalar with respect to a (column) vector is a (row) vector consisting of the derivatives of the scalar w.r.t. the components of the vector, and the derivative of a vector w.r.t. a scalar is the obvious vector of derivatives.

Usually, matrix calculus assumes real-valued vectors, but the extension to complex vectors via the Wirtinger derivatives discussed above is a relatively straightforward. The use of [Dirac ("braket") notation](https://en.wikipedia.org/wiki/Bra–ket_notation) helps tremendously here: ``|\Psi_k⟩`` describes a complex column vector, and ``⟨\Psi_k|`` describes the corresponding row vector with complex-conjugated elements. These can take the place of ``z_k`` and ``z_k^*`` in the Wirtinger derivative. Consider, e.g.,

```math
\begin{equation}\label{eq:Jsm}
J(\{|\Psi_k⟩\})
= \sum_k \vert \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \vert^2
= \sum_k \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \langle \Psi_k^{\text{tgt}} \vert \Psi_k \rangle\,,
\end{equation}
```

for a fixed set of "target states" ``|\Psi_k^{\text{tgt}}⟩``.

The derivative ``\partial J/\partial |\Psi_k⟩`` is

```math
\begin{equation}\label{eq:dJ_dKet}
\frac{\partial J}{\partial |\Psi_k}⟩ = \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \langle\Psi_k^{\text{tgt}}\vert\,,
\end{equation}
```

in the same sense as Eq. \eqref{eq:wirtinger1}. We simply treat ``|\Psi_k⟩`` and ``⟨\Psi_k|`` as independent variables corresponding to ``z_k`` and ``z_k^*``. Note that the result is a "bra", that is, a co-state, or _row vector_. The braket notation resolves the question of ["layout conventions"](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions) in matrix calculus in favor of the "numerator layout". Consequently, we also have a well-defined derivative w.r.t. the co-state:

```math
\begin{equation}
\frac{\partial J}{\partial ⟨\Psi_k|} = \langle \Psi_k^{\text{tgt}} \vert \Psi_k \rangle \vert\Psi_k^{\text{tgt}}\rangle\,,
\end{equation}
```

which we can either get explicitly from Eq. \eqref{eq:Jsm}, differentiating w.r.t. ``|\Psi_k⟩`` as an independent parameter and changing the order of the factors, or implicitly by taking the conjugate transpose of Eq. \eqref{eq:dJ_dKet}.

For the full chain rule of a functional ``J(\{|\Psi_k(\{\epsilon_{nl}\})⟩\})``, we thus find


```math
\begin{equation}\label{eq:grad-via-chi1}
  (\nabla J)_{nl}
  \equiv \frac{\partial J}{\partial \epsilon_{nl}}
  = 2 \Re \sum_k \left(
    \frac{\partial J}{\partial |\Psi_k⟩}
    \frac{\partial |\Psi_k⟩}{\partial \epsilon_{nl}}
  \right)\,.
\end{equation}
```

With the definition in Eq. \eqref{eq:wirtinger1}, this corresponds directly to the scalar

```math
\begin{equation}
  \frac{\partial J}{\partial \epsilon_{nl}}
  = \sum_{km} \left(
    \frac{\partial J}{\partial \Re[\Psi_{km}]}
    \frac{\partial \Re[\Psi_{km}]}{\partial \epsilon_{nl}}
    + \frac{\partial J}{\partial \Im[\Psi_{km}]}
    \frac{\partial \Im[\Psi_{km}]}{\partial \epsilon_{nl}}
  \right)\,,
\end{equation}
```

where the complex scalar ``\Psi_{km}`` is the ``m``'th element of the ``k``'th vector, and corresponds to the ``z_k`` in Eq. \eqref{eq:wirtinger_chainrule}.

In open quantum systems, where the state is described by a density matrix ``\hat{\rho}``, it can be helpful to adopt the double-braket notation ``\langle\!\langle \hat{\rho}_1 \vert \hat{\rho}_2 \rangle\!\rangle \equiv \tr[\hat{\rho}_1^\dagger \hat{\rho}_2]``, respectively to keep track of normal states ``\hat{\rho}`` (corresponding to ``|\Psi⟩``) and adjoint states ``\hat{\rho}^\dagger`` (corresponding to ``⟨\Psi|``), even when ``\hat{\rho}`` is Hermitian and thus ``\hat{\rho} = \hat{\rho}^\dagger``. For numerical purposes, density matrices are best vectorized by concatenating the columns of ``\hat{\rho}`` into a single column vector ``\vec{\rho}``. Thus, we do not have be concerned with a separate definition of derivatives w.r.t. density matrices.


## Gradients for final-time functionals

For simplicity, we consider a functional defined entirely at final time ``T``, the ``J_T`` term in Eq. \eqref{eq:grape-functional}. Since ``J_T`` depends explicitly on ``\{|\Psi_k(T)⟩\}`` and only implicitly on ``\{\epsilon_{nl}\}``, we can use the complex chain rule in Eq. \eqref{eq:grad-via-chi1}.

Further, we define a new state

```math
\begin{equation}\label{eq:chi}
|\chi_k(T)⟩ \equiv - \frac{\partial J_T}{\partial ⟨\Psi_k(T)|}
\end{equation}
```

The minus sign in this definition is arbitrary, and is intended solely to match an identical definition in [Krotov's method](@extref Krotov :doc:`index`), the most direct alternative to GRAPE. Since ``|\chi_k(T)⟩`` does not depend on ``\epsilon_{nl}``, we can pull forward the derivative ``\partial / \partial \epsilon_{nl}`` in Eq. \eqref{eq:grad-via-chi1}, writing it as

```math
\begin{equation}\label{eq:grad-at-T}
(\nabla J_T)_{nl}
= \frac{\partial J_T}{\partial \epsilon_{nl}}
= - 2 \Re \sum_k \frac{\partial}{\partial \epsilon_{nl}} \langle \chi_k(T) \vert \Psi_k(T)\rangle\,.
\end{equation}
```

We end up with the gradient of $J_T$ being the derivative of the overlap of two states ``|\chi_k(T)⟩`` and ``|\Psi_k(T)⟩`` at final time ``T``.

Next, we make use the assumption that the time evolution is piecewise constant, so that we can use the time evolution operator defined in Eq. \eqref{eq:time-evolution-op} to write ``|\Psi_k(T)⟩`` as the time evolution of an initial state ``\Psi_k(t=0)``, the `initial_state` of the ``k``'th [`trajectory`](@extref `QuantumControl.Trajectory`) in the [`QuantumControl.ControlProblem`](@extref). That is, ``|\Psi_k(T)⟩ = \hat{U}^{(k)}_{N_T} \dots \hat{U}^{(k)}_1 |\Psi_k(t=0)⟩`` with the time evolution operator ``\hat{U}^{(k)}_n`` for the ``n``'th time interval of the time grid with ``N_T + 1`` time grid points, cf. Eq. \eqref{eq:time-evolution-op}. Plugging this into Eq. \eqref{eq:grad-at-T} immediately gives us

```math
\begin{equation}\label{eq:grad-at-T-U}
\begin{split}
\frac{\partial J_T}{\partial \epsilon_{nl}}
&= -2 \Re \sum_k \frac{\partial}{\partial \epsilon_{nl}}
    \bigg\langle \chi_k(T) \bigg\vert \hat{U}_{N_T}^{(k)} \dots \hat{U}^{(k)}_n \dots \hat{U}^{(k)}_1 \bigg\vert \Psi_k(t=0) \bigg\rangle
    \\
&= -2 \Re \sum_k \bigg\langle \chi_k(t_{n}) \bigg\vert \frac{\partial \hat{U}^{(k)}_n}{\partial \epsilon_{nl}} \bigg\vert \Psi_k(t_{n-1}) \bigg\rangle
\end{split}
\end{equation}
```

with $|\chi_k(t_{n})⟩ = U^{\dagger (k)}_{n+1} \dots U^{\dagger(k)}_{N_T} |\chi_k(T)⟩$, i.e., a backward-propagation of the state given by Eq. \eqref{eq:chi} with the adjoint Hamiltonian or Liouvillian and $|\Psi_k(t_{n-1})⟩ = \hat{U}^{(k)}_{n-1}\dots \hat{U}^{(k)}_1 |\Psi_k(0)⟩$, i.e., a forward-propagation of the initial state of the ``k``'th trajectory.



## Derivative of the time-evolution operator

The last missing piece for evaluating the gradient in Eq. \eqref{eq:grad-at-T-U} is the derivative of the time evolution operator ``\hat{U}_n^{(k)}`` for the current time interval ``n``. The operator ``\frac{\partial \hat{U}_n^{(k)}}{\partial \epsilon_{nl}}`` could either act to the right, being applied to ``|\Psi_k(t_{n-1})⟩`` during the forward propagation, or it (or rather it's conjugate transpose) could act to the left, being applied to ``|\chi_k(t_n)⟩`` during the backward propagation. For reasons that will be explained later on, it is numerically more efficient to include it in the backward propagation. Thus, we are given a state ``|\chi_k(t_{n})⟩`` and must then numerically obtain the state

```math
\begin{equation}\label{eq:U-deriv}
|\chi^\prime_{kl}(t_{n-1})⟩
\equiv  \frac{\partial \hat{U}^{\dagger(k)}_n}{\partial \epsilon_{nl}} |\chi_k(t_n)⟩
= \frac{\partial}{\partial \epsilon_{nl}} \exp\left[-\ii \hat{H}^{\dagger}_{k}(\{\epsilon_{nl}\}) dt^{(-)}_n \right] |\chi_k(t_n)⟩\,.
\end{equation}
```

Note the dagger and the negative time step ``dt^{(-)}_n = (t_{n-1} - t_{n})`` — in lieu of changing the sign of the imaginary unit ``\ii`` — to account for the fact that we are doing a backward-propagation, cf. the corresponding forward-propagation in Eq. \eqref{eq:time-evolution-op}. Of course, for a standard Schrödinger equation, ``\hat{H}_{kn}^\dagger = \hat{H}_{kn}``, and then the negative time step is the only difference between backward and forward propagation; but, in general, we also allow for non-Hermitian Hamiltonians or Liouvillians where it is important to use the correct (adjoint) operator.

Thus, Eq. \eqref{eq:grad-at-T-U} turns into

```math
\begin{equation}\label{eq:grad-via-chi-prime}
\frac{\partial J_T}{\partial \epsilon_{nl}}
= -2 \Re \sum_k \bigg \langle \chi^\prime_{kl}(t_{n-1}) \bigg\vert \Psi_k(t_{n-1}) \bigg \rangle\,.
\end{equation}
```

Or, equivalently, if we had let ``\frac{\partial \hat{U}_n^{(k)}}{\partial \epsilon_{nl}}`` act to the right,

```math
\begin{equation}\label{eq:grad-via-psi-prime}
\frac{\partial J_T}{\partial \epsilon_{nl}}
= -2 \Re \sum_k \bigg \langle \chi_{kl}(t_{n}) \bigg\vert \Psi^{\prime}_k(t_{n}) \bigg \rangle\,.
\end{equation}
```

with ``|\Psi^{\prime}_k(t_{n})⟩ \equiv  \frac{\partial \hat{U}^{(k)}_n}{\partial \epsilon_{nl}} |\Psi_k(t_{n-1})⟩``.

### [Taylor expansion](@id Overview-Taylor)

There are several possibilities for evaluating Eq. \eqref{eq:U-deriv}. One method is to expand the exponential into a Taylor series [KuprovJCP2009; Eq. (20)](@cite)

```math
\begin{equation}\label{eq:taylor-op}
\frac{\partial \hat{U}^{\dagger(k)}_n}{\partial \epsilon_{nl}}
= \sum_{m=1}^{\infty} \frac{\left(-\ii \hat{H}^{\dagger}_{kn} dt^{(-)}_n\right)^m}{m!}
    \sum_{m^\prime=0}^{m-1}
    {\hat{H}^{\dagger}_{kn}}^{\!\!m^\prime}
    \hat{\mu}_{lkn}^{\dagger}
    {\hat{H}^{\dagger}_{kn}}^{\!\!m-m^\prime-1}
\end{equation}
```

with ``\hat{H}_{kn} \equiv \hat{H}_{k}(\{\epsilon_{nl}\})`` and ``\hat{\mu}_{lkn} \equiv \frac{\partial \hat{H}_{kn}}{\partial \epsilon_{nl}}``.

In practice, Eq. \eqref{eq:taylor-op} is best evaluated recursively, while  being applied to  ``|\chi_k(t_n)⟩``:

```math
\begin{equation}
\ket{\chi^\prime_{kl}(t_{n-1})} = \sum_{m=1}^{\infty} \frac{\left(-\ii \, dt_n^{(-)}\right)^m}{m!} \ket{\Phi^{(lkn)}_m}\,,
\end{equation}
```

with

```math
\begin{equation}
\begin{split}
  \ket{\Phi^{(lkn)}_1} &= \hat{\mu}_{lkn}^{\dagger} \ket{\chi_k(t_n)}\,,              \\
  \ket{\Phi^{(lkn)}_m} &= \hat{\mu}_{lkn}^{\dagger} {\hat{H}^{\dagger}_{kn}}^{\!\!m-1}  \ket{\chi_k(t_n)} + {\hat{H}^{\dagger}_{kn}} \ket{\Phi^{(lkn)}_{m-1}}\,.
\end{split}
\end{equation}
```

In `GRAPE.jl`, Eq. \eqref{eq:U-deriv} can be evaluated via a Taylor expansion as described above by passing `gradient_method=:taylor`, with further options to limit the maximum order ``m``.

!!! tip "TMIDR"

    As in the [general TMIDR](#tmidr), the indices ``k`` and ``l`` are somewhat superfluous here. In addition, ``\hat{\mu}_{lkn} \equiv \frac{\partial \hat{H}_{kn}}{\partial \epsilon_{nl}}`` still depends on ``\epsilon_{nl}`` only for non-linear controls. Much more commonly, , for linear Hamiltonians of the form ``\hat{H} = \hat{H_0} + \epsilon(t) \hat{\mu}``, ``\hat{\mu}`` is just a static [control operator](@extref QuantumControl :label:`Control-Operator`). If ``\hat{H}`` is a standard Hamiltonian, and thus Hermitian, we can drop the dagger. The time gid is usually uniform, so we can drop the index ``n`` from ``dt``. Thus, a simplified version of Eq. \eqref{eq:taylor-op} is

    ```math
    \begin{equation}\label{eq:taylor-op-simplified}
    \frac{\partial \hat{U}^{\dagger}_n}{\partial \epsilon_{n}}
    = \sum_{m=1}^{\infty} \frac{\left(-\ii \hat{H}_{n} dt^{(-)}\right)^m}{m!}
        \sum_{m^\prime=0}^{m-1}
        \hat{H}_{n}^{m^\prime}
        \hat{\mu}
        {\hat{H}}_{n}^{m-m^\prime-1}\,,
    \end{equation}
    ```

    with the recursive formula

    ```math
    \begin{equation}
    \begin{split}
    \ket{\chi^\prime(t_{n-1})} &= \sum_{m=1}^{\infty} \frac{\left(-\ii \, dt^{(-)}\right)^m}{m!} \ket{\Phi_m}\,,\\
    \ket{\Phi_1} &= \hat{\mu} \ket{\chi_k(t_n)}\,,              \\
    \ket{\Phi_m} &= \hat{\mu} {\hat{H}_{n}}^{\!\!m-1}  \ket{\chi_k(t_n)} + {\hat{H}_{n}} \ket{\Phi_{m-1}}\,.
    \end{split}
    \end{equation}
    ```

For sufficiently small time steps,  one may consider using only the first term in the Taylor series, ``|\chi^\prime_{kl}(t_{n-1})⟩  \approx -\ii dt_n^{(-)} |\Phi^{(lkn)}_1⟩``. That is, from Eq. \eqref{eq:grad-via-chi-prime}, we get

```math
\begin{equation}
\begin{split}
\frac{\partial J_T}{\partial \epsilon_{nl}}
&\approx 2\,dt_n\,\Im \sum_k \bigg \langle \chi_{kl}(t_{n}) \bigg\vert \frac{\partial \hat{H}_{kn}}{\partial \epsilon_{nl}} \bigg\vert \Psi_k(t_{n-1}) \bigg \rangle \\
&\approx 2\,dt_n\,\Im \sum_k \bigg \langle \chi_{kl}(t_{n}) \bigg\vert \frac{\partial \hat{H}_{kn}}{\partial \epsilon_{nl}} \bigg\vert \Psi_k(t_{n}) \bigg \rangle\,.
\end{split}
\end{equation}
```

This approximation of the gradient has been used historically, including in GRAPE's original formulation [KhanejaJMR2005](@cite), also because it matches optimality conditions derived in a Lagrange-multiplier formalism [PeircePRA1988; BorziPRA2002](@cite) that pre-dates GRAPE. The derivation via Lagrange multipliers also extends more easily to equations of motion beyond Eq. \eqref{eq:tdse} such as Gross–Pitaevskii equation [HohenesterPRA2007, JaegerPRA2014](@cite). However, even though it is considered a "gradient-type" optimization, it is not considered to be within the scope of the `GRAPE` package (up to the ability to limit that Taylor expansion to first order). The conceptual difference is that these older methods (as well as other "gradient-type" [Krotov's method](@extref Krotov :doc:`index`)) derive optimality conditions _first_ (via functional derivatives), and the add time discretization to arrive at a numerical scheme. In contrast, `GRAPE` discretizes _first_, and then obtains gradients via simple derivatives w.r.t. the pulse values ``\epsilon_{nl}``. This concept of "discretize first" is _the_ core concept exploited in `GRAPE.jl`.

After GRAPE's original formulation [KhanejaJMR2005](@cite), it was quickly realized that high-precision gradients are essential for numerical stability and convergence, in particular if the gradient is then used in a quasi-Newton method [KuprovJCP2009, FouquieresJMR2011](@cite). Thus, low-order Taylor expansions should be avoided in most contexts.

### [Gradient Generators](@id Overview-Gradgen)

In order to evaluate Eq. \eqref{eq:U-deriv} to high precision, one can use a trick from computational linear algebra [VanLoanITAC1978](@cite) that was reformulated in the context of quantum control by [GoodwinJCP2015](@citet). It allows to calculate ``|\chi^\prime_{kl}(t_{n-1})⟩ \equiv  \frac{\partial \hat{U}^{\dagger(k)}_n}{\partial \epsilon_{nl}} |\chi_k(t_n)⟩`` and ``|\chi_{kl}(t_{n-1})⟩ = \hat{U}^{\dagger(k)}_n |\chi_k(t_n)⟩`` at the same time, as

```math
\begin{equation}\label{eq:gradprop-bw}
  \begin{pmatrix} \ket{\chi^{\prime}_{k1}(t_{n-1})} \\ \vdots \\ \ket{\chi^{\prime}_{kL}(t_{n-1})} \\ \ket{\chi_k(t_{n-1})} \end{pmatrix}
  = \exp \left[-\ii\,G[\hat{H}_{kn}^{\dagger}]\,dt_n\right] \ket{\tilde\chi_k(t_n)}
  \,,\\
\end{equation}
```

by backward-propagating an extended state

```math
\begin{equation}\label{eq:gradgen-state}
    \ket{\tilde\chi_k(t_n)}
    \equiv \begin{pmatrix} 0 \\ \vdots \\ 0 \\ \ket{\chi_k(t_n)} \end{pmatrix}
\end{equation}
```

of dimension ``N(L+1)``, where ``L`` is the number of controls and ``N`` is the dimension of ``|\chi_k⟩``, under a "gradient generator"


```math
\begin{equation}\label{eq:gradgen}
    G[\hat{H}_{kn}^{\dagger}]
    = \begin{pmatrix}
        \hat{H}^\dagger_{kn} & 0 & \dots & 0 &\hat{\mu}_{1kn}^{\dagger} \\
        0 & \hat{H}^\dagger_{kn} & \dots & 0 & \hat{\mu}_{2kn}^{\dagger} \\
        \vdots & & \ddots & & \vdots \\
        0 & 0 & \dots & \hat{H}^\dagger_{kn} & \hat{\mu}_{Lkn}^{\dagger} \\
        0 & 0 & \dots & 0 & \hat{H}^\dagger_{kn}
    \end{pmatrix}\,.
\end{equation}
```

This is a purely formal way of writing the gradient generator; in practice, the extended state ``\ket{\tilde\chi_k(t_n)}`` is represented by a data structure with slots for the states ``|\chi^{\prime}_{1}⟩`` … ``|\chi^{\prime}_{L}⟩``, in addition o the original state ``|\chi⟩``, and ``G[\hat{H}_{kn}^{\dagger}]`` is a container around all the operators ``\hat{\mu}^{\dagger}_{lkn} \equiv \frac{\partial \hat{H}^{\dagger}_{kn}}{\partial \epsilon_{nl}}`` in addition to the original (adjoint) Hamiltonian or Liouvillian ``\hat{H}^{\dagger}_{kn}`` itself. The gradient generator ``G`` is then implicitly defined by how it acts on the extended state,

```math
\begin{equation}
    G[\hat{H}] \begin{pmatrix}
        \ket{\chi^{\prime}_{1}} \\ \vdots \\ \ket{\chi^{\prime}_{L}} \\ \ket{\chi}
    \end{pmatrix}
    = \begin{pmatrix}
        \hat{H} \ket{\chi^{\prime}_{1}} + \hat{\mu}_1 \ket{\chi} \\
        \vdots \\
        \hat{H} \ket{\chi^{\prime}_{L}} + \hat{\mu}_L \ket{\chi} \\
        \hat{H} \ket{\chi}
    \end{pmatrix}\,.
\end{equation}
```

This way, ``G[\hat{H}^{\dagger}_{kn}]`` replaces ``\hat{H}^{\dagger}_{kn}`` in the backwards propagation, using any of the methods in [`QuantumPropagators`](@extref QuantumPropagators :doc:`index`), e.g., the polynomial Chebychev propagator. The appropriate data structures for ``G`` and the extended states are implemented in [the `QuantumGradientGenerators` package](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl). As it provides an "exact" gradient independently of the time step, the use of the gradient generator is the default in `GRAPE.jl`, or it can be explicitly requested with `gradient_method=:gradgen`.


## GRAPE scheme

With Eq. \eqref{eq:grad-at-T-U} and the [use of gradient generators](@ref Overview-Gradgen) explained above, we end up wht an efficient numerical scheme for evaluating the full gradient shown in [Fig. 1](#fig-grape-scheme).

```@raw html
<p id="fig-grape-scheme" style="text-align: center">
<a href="../fig/grape_scheme.png">
<img src="../fig/grape_scheme.png" width="100%"/>
</a>
<a href="#fig-grape-scheme">Figure 1</a>: Numerical scheme for the evaluation of the gradient in `GRAPE.jl`.
</p>
```

In each iteration, we start in the bottom left with the initial state ``|\Psi_k(t=t_0=0)⟩`` for the ``k'th`` trajectory. This state is forward-propagated under the guess pulse (in parallel for the different trajectories) over the entire time grid until final time ``T`` with ``N_T`` time steps. In the diagram, ``t_{-n}`` is a shorthand for ``t_{N_T - n}``. The propagation over the ``n``'th time interval uses the pulse values ``\epsilon_{nl}``. In the diagram, we have omitted the index ``l`` for the different control functions ``\epsilon_l(t)`` ([TMIDR](#tmidr)). All of the forward-propagated states (red in the diagram) must be stored in memory.

Having determined ``|\Psi_k(T)⟩``, the state ``\ket{\chi_k(T)}`` is calculated according to Eq. \eqref{eq:chi} for each trajectory ``k``. With the default `gradient_method=:gradgen` that is depicted here, ``\ket{\chi_k(T)}`` is then converted into a zero-padded extended state ``|\tilde\chi_k(T)⟩``, see Eq. \eqref{eq:gradgen-state}, which is then backward propagated under a gradient-generator ``G[\hat{H}_{kn}^{\dagger}]`` defined according to Eq. \eqref{eq:gradgen}.

After each step in the backward propagation, the extended state ``|\tilde\chi_k(t_n)\rangle⟩`` contains the gradient-states ``|\chi^{\prime}_{kl}(t_n)⟩``, cf. Eq. \eqref{eq:gradprop-bw}. The corresponding forward-propagated states ``\Psi_k(t_n)`` are read from storage; the overlap
``⟨\chi^{\prime}_{kl}(t_n)|\Psi_k(t_n)⟩`` then contributes to the element ``(\nabla J)_{nl}`` of the gradient, cf. Eq. \eqref{eq:grad-via-chi-prime}.

In the original formulation of GRAPE [KhanejaJMR2005](@cite), ``|\chi_k(T)⟩`` is always the target state associated with the ``k``'th trajectory. This makes it arbitrary whether to forward-propagated (and store) ``|\Psi_k(t)⟩`` first, or backward-propagate (and store) ``|\chi_k(t)⟩`` first. However, with the generalization to arbitrary functionals [GoerzQ2022](@cite) via the definition in Eq. \eqref{eq:chi}, ``|\chi_k(T)⟩`` can now depend on the forward-propagates states ``\{|\Psi_k(T)⟩\}``. Thus, the forward propagation and storage must always precede the backward propagation. The requirement for storing the forward-propagated states also explains the choice to let ``\frac{\partial \hat{U}_n^{(k)}}{\partial \epsilon_{nl}}`` act to the left in Eq. \eqref{eq:grad-at-T-U} to get Eq. \eqref{eq:grad-via-chi-prime}. If we had instead chose to let the derivative act to the right to get Eq. \eqref{eq:grad-via-psi-prime}, we would have to store all of the states ``|\Psi^{\prime}_k(t_{n})⟩`` in addition to just ``|\Psi_k(t_{n})⟩`` for every time step, which would increase the required memory ``L``-fold for ``L`` controls.

The above scheme may be further augmented for [running costs](@ref Overview-Running-Costs). Also, if the alternative `gradient_method=:taylor` is used, the backward propagation is of the normal states ``|\chi_k(T)⟩`` instead of the extended ``|\chi_k(T)⟩``, but the ``|\chi^{\prime}_{kl}(t_n)⟩`` still have to be evaluated in each time step. In any case, once the full gradient vector has been collected, it is passed to an [optimizer such as L-BFGS-B](@ref Optimizers).


## [Semi-automatic differentiation](@id Overview-SemiAD)

Same as GRAPE, up to definition of chi. Special cases for overlap functionals and gate functionals.

How "gradients" are implemented in Zygote.


## [Running costs](@id Overview-Running-Costs)

## Optimizers

Once the gradient has been evaluated, in the original formulation of GRAPE [KhanejaJMR2005](@cite), the values ``\epsilon_{nl}`` would then be updated by taking a step with a fixed step width ``\alpha`` in the direction of the negative gradient, to iteratively minimize the value of the optimization functional ``J``. In practice, the gradient can also be fed into an arbitrary gradient-based optimizer, and in particular a quasi-Newton method like L-BFGS-B [ZhuATMS1997, LBFGSB.jl](@cite). This results in a dramatic improvement in stability and convergence [FouquieresJMR2011](@cite), and is assumed as the default in `GRAPE.jl`. Gradients of the time evolution operator can be evaluated to machine precision following [GoodwinJCP2015](@citet). The GRAPE method could also be extended to a true Hessian of the optimization functional [GoodwinJCP2016](@cite), which would be in scope for future versions of `GRAPE.jl`.



* gradient descent
* L-BFGS-B
* bounded controls
* `Optim.jl`
