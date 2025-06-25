# Background

The GRAPE methods minimizes an optimization functional of the form

```math
\begin{equation}\label{eq:grape-functional}
J(\{ϵ_l(t)\})
    = J_T(\{|Ψ_k(T)⟩\})
    + λ_a \, \underbrace{∑_l \int_{0}^{T} g_a(ϵ_l(t)) \, dt}_{=J_a(\{ϵ_l(t)\})}
    + λ_b \, \underbrace{∑_k \int_{0}^{T} g_b(|Ψ_k(t)⟩) \, dt}_{=J_b(\{|Ψ_k(t)⟩\})}\,,
\end{equation}
```

where ``\{ϵ_l(t)\}`` is a set of [control functions](@extref QuantumControl :label:`Control-Function`) defined between the initial time ``t=0`` and the final time ``t=T``, and ``\{|Ψ_k(t)⟩\}`` is a set of ["trajectories"](@extref QuantumControl.Trajectory) evolving from a set of initial states ``\{|\Psi_k(t=0)⟩\}`` under the controls ``\{ϵ_l(t)\}``. The primary focus is on the final-time functional ``J_T``, but running costs ``J_a`` (weighted by ``λ_a``) may be included penalize certain features of the control field. In principle, a state-dependent running cost ``J_b`` weighted by ``λ_b`` can also be included (and will be discussed below), although this is currently not fully implemented in `GRAPE.jl`.

The defining assumptions of the GRAPE method are

1.  The control fields ``\epsilon_l(t)`` are piecewise-constant on the intervals of a time grid. That is, we have a vector of pulse values with elements ``\epsilon_{nl}``. We use the double-index `nl`, for the value of the ``l``'th control field on the ``n``'th interval of the time grid.

2. The states ``\ket{\Psi_k(t)}`` evolve under an equation of motion of the form

```math
\begin{equation}\label{eq:tdse}
    i \hbar \frac{\partial \ket{\Psi_k(t)}}{\partial t} = \hat{H}_k(\{\epsilon_l(t)\}) \ket{\Psi_k(t)}\,.
\end{equation}
```

This includes the Schrödinger equation, but also the Liouville equation for open quantum systems. In the latter case ``\ket{\Psi_k}`` is replaced by a vectorized density matrix, and ``\hat{H}_k`` is replaced by a Liouvillian (super-) operator describing the dynamics of the ``k``'th trajectory. The crucial point is that Eq. \eqref{eq:tdse} can be solved analytically within each time interval as

```math
\begin{equation}\label{eq:time-evolution-op}
    \ket{\Psi_k(t_{n+1})} = \underbrace{\exp\left[-i \hat{H}_{kn} dt_n \right]}_{=\hat{U}_{kn}} \ket{\Psi_k(t_n)}\,.
\end{equation}
```

These two assumptions allow to analytically derive the gradient ``(\nabla J)_{nl} \equiv \frac{\partial J}{\partial \epsilon_{nl}}``. The initial derivation of GRAPE by [KhanejaJMR2005](@citet) focuses on a final-time functional ``J_T`` that depends of the overlap of each forward-propagated ``\ket{\Psi_k(T)}`` with a target state ``\ket{\Psi^{\text{tgt}}_k(T)}`` and updates the pulse values ``\epsilon_{nl}`` directly in the direction of the negative gradient. Improving on this, [FouquieresJMR2011](@citet) showed that using a quasi-Newton method to update the pulses based on the gradient information leads to a dramatic improvement in convergence and stability. Furthermore, [GoodwinJCP2015](@citet) improved on the precision of evaluating the gradient of a local time evolution operator, which is a critical step in the GRAPE scheme. Finally, [GoerzQ2022](@citet) generalized GRAPE to arbitrary functionals of the form \eqref{eq:grape-functional}, bridging the gap to automatic differentiation techniques [LeungPRA2017, AbdelhafezPRA2019, AbdelhafezPRA2020](@cite) by introducing the technique of "semi-automatic differentiation". This most general derivation is the basis for the implementation in `GRAPE.jl`, and is reproduced below.


## Prerequisite: Wirtinger derivatives and matrix calculus

Even though we are seeking the derivative of the real-valued functional ``J`` with respect to the real-valued parameter ``\epsilon_{nl}``, the functional still involves complex quantities via ``\ket{\Psi_k(t)}`` and ``\hat{H}`` in Eq. \eqref{eq:tdse}. In order to apply the chain rule in the derivation of the gradient, we will have to clarify the notion of derivatives in the context of complex numbers, as well as derivatives with respect to vectors ("matrix calculus").

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
\def\ii{\mathrm{i}}
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

We can now go one step further and allow for intermediate variables that are complex _vectors_ instead of scalars, ``J(\{\epsilon_{nl}\}) \rightarrow J(\{\ket{\Psi_k(\{\epsilon_{nl}\})}\})``. Taking the derivative w.r.t. a vector puts us in the domain of [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus). Fundamentally, the derivative of a scalar with respect to a (column) vector is a (row) vector consisting of the derivatives of the scalar w.r.t. the components of the vector, and the derivative of a vector w.r.t. a scalar is the obvious vector of derivatives.

Usually, matrix calculus assumes real-valued vectors, but the extension to complex vectors via the Wirtinger derivatives discussed above is a relatively straightforward. The use of [Dirac ("braket") notation](https://en.wikipedia.org/wiki/Bra–ket_notation) helps tremendously here: ``\ket{\Psi_k}`` describes a complex column vector, and ``\bra{\Psi_k}`` describes the corresponding row vector with complex-conjugated elements. These can take the place of ``z_k`` and ``z_k^*`` in the Wirtinger derivative. Consider, e.g.,

```math
\begin{equation}\label{eq:Jsm}
J(\{\ket{\Psi_k}\})
= \sum_k \vert \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \vert^2
= \sum_k \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \langle \Psi_k^{\text{tgt}} \vert \Psi_k \rangle\,,
\end{equation}
```

for a fixed set of "target states" ``\ket{\Psi_k^{\text{tgt}}}``.

The derivative ``\partial J/\partial \ket{\Psi_k}`` is

```math
\begin{equation}\label{eq:dJ_dKet}
\frac{\partial J}{\partial \ket{\Psi_k}} = \langle \Psi_k \vert \Psi_k^{\text{tgt}} \rangle \langle\Psi_k^{\text{tgt}}\vert\,,
\end{equation}
```

in the same sense as Eq. \eqref{eq:wirtinger1}. We simply treat ``\ket{\Psi_k}`` and ``\bra{\Psi_k}`` as independent variables corresponding to ``z_k`` and ``z_k^*``. Note that the result is a "bra", that is, a co-state, or _row vector_. The braket notation resolves the question of ["layout conventions"](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions) in matrix calculus in favor of the "numerator layout". Consequently, we also have a well-defined derivative w.r.t. the co-state:

```math
\begin{equation}
\frac{\partial J}{\partial \bra{\Psi_k}} = \langle \Psi_k^{\text{tgt}} \vert \Psi_k \rangle \vert\Psi_k^{\text{tgt}}\rangle\,,
\end{equation}
```

which we can either get explicitly from Eq. \eqref{eq:Jsm}, differentiating w.r.t. ``\ket{\Psi_k}`` as an independent parameter and changing the order of the factors, or implicitly by taking the conjugate transpose of Eq. \eqref{eq:dJ_dKet}.

For the full chain rule of a functional ``J(\{\ket{\Psi_k(\{\epsilon_{nl}\})}\})``, we thus find


```math
\begin{equation}\label{eq:grad-via-chi1}
  (\nabla J)_{nl}
  \equiv \frac{\partial J}{\partial \epsilon_{nl}}
  = 2 \Re \sum_k \left(
    \frac{\partial J}{\partial \ket{\Psi_k}}
    \frac{\partial \ket{\Psi_k}}{\partial \epsilon_{nl}}
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


## Gradients for final-time functionals

For simplicity, we consider a functional defined entirely at final time ``T``, the ``J_T`` term in Eq. \eqref{eq:grape-functional}. Since ``J_T`` depends explicitly on ``\{\ket{\Psi_k(T)}\}`` and only implicitly on ``\{\epsilon_{nl}\}``, we can use the complex chain rule in Eq. \eqref{eq:grad-via-chi1}.

Further, we define a new state

```math
\begin{equation}\label{eq:chi}
\ket{\chi_k(T)} \equiv - \frac{\partial J_T}{\partial \bra{\Psi_k(T)}}
\end{equation}
```

The minus sign in this definition is arbitrary, and is intended solely to match an identical definition in [Krotov's method](@extref Krotov :doc:`index`), the most direct alternative to GRAPE. Since ``\ket{\chi_k(T)}`` does not depend on ``\epsilon_{nl}``, we can pull forward the derivative ``\partial / \partial \epsilon_{nl}`` in Eq. \eqref{eq:grad-via-chi1}, writing it as

```math
\begin{equation}
(\nabla J_T)_{nl}
= \frac{\partial J_T}{\partial \epsilon_{nl}}
= - 2 \Re \sum_k \frac{\partial}{\partial \epsilon_{nl}} \Braket{\chi_k(T) | \Psi_k(T)}\,.
\end{equation}
```

We end up with the gradient of $J_T$ being the derivative of the overlap of two states ``\ket{\chi_k(T)}`` and ``\ket{\Psi_k(T)}`` at final time ``T``.

Next, we make use the assumption that the time evolution is piecewise constant, so that we can use the time evolution operator defined in Eq. \eqref{eq:time-evolution-op} to write ``\ket{\Psi_k(T)}`` as the time evolution of an initial state ``\Psi_k(t=0)``, the `initial_state` of the ``k``'th [`trajectory`](@extref `QuantumControl.Trajectory`) in the [`QuantumControl.ControlProblem`](@extref).


## Derivative of the time-evolution operator

* Taylor
* Schirmer-gradient
* Comment on first-order Taylor, Lagrange multipliers, Gross–Pitaevskii equation

## GRAPE scheme

It results in an efficient numerical scheme for evaluating the full gradient [GoerzQ2022; Figure 1(a)](@cite). The scheme extends to situations where the functional is evaluated on top of *multiple* propagated states ``\{\vert \Psi_k(t) \rangle\}`` with an index ``k``, and multiple controls ``\epsilon_l(t)``, resulting in a vector of values ``\epsilon_{nl}`` with a double-index ``nl``. Once the gradient has been evaluated, in the original formulation of GRAPE [KhanejaJMR2005](@cite), the values ``\epsilon_{nl}`` would then be updated by taking a step with a fixed step width ``\alpha`` in the direction of the negative gradient, to iteratively minimize the value of the optimization functional ``J``. In practice, the gradient can also be fed into an arbitrary gradient-based optimizer, and in particular a quasi-Newton method like L-BFGS-B [ZhuATMS1997, LBFGSB.jl](@cite). This results in a dramatic improvement in stability and convergence [FouquieresJMR2011](@cite), and is assumed as the default in `GRAPE.jl`. Gradients of the time evolution operator can be evaluated to machine precision following [GoodwinJCP2015](@citet). The GRAPE method could also be extended to a true Hessian of the optimization functional [GoodwinJCP2016](@cite), which would be in scope for future versions of `GRAPE.jl`.


```@raw html
<p id="fig-grape-scheme" style="text-align: center">
<a href="../fig/grape_scheme.png">
<img src="../fig/grape_scheme.png" width="100%"/>
</a>
<a href="#fig-grape-scheme">Figure 1</a>: Numerical scheme for the evaluation of the gradient in GRAPE, with semi-automatic differentiation
</p>
```

See the scheme depicted in [Fig. 1](#fig-grape-scheme).

## Semi-automatic differentiation

Same as GRAPE, up to definition of chi. Special cases for overlap functionals and gate functionals.


## Running costs

## Comparison with Krotov's method
