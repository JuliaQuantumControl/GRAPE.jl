```@meta
CurrentModule = GRAPE
```

# GRAPE.jl

```@eval
using Markdown
using Pkg

VERSION = Pkg.dependencies()[Base.UUID("6b52fcaf-80fe-489a-93e9-9f92080510be")].version

github_badge = "[![Github](https://img.shields.io/badge/JuliaQuantumControl-GRAPE.jl-blue.svg?logo=github)](https://github.com/JuliaQuantumControl/GRAPE.jl)"

version_badge = "![v$VERSION](https://img.shields.io/badge/version-v$VERSION-green.svg)"

Markdown.parse("$github_badge $version_badge")
```

Gradient Ascent Pulse Engineering in Julia

## Summary

The `GRAPE.jl` package implements Gradient Ascent Pulse Engineering [KhanejaJMR2005](@cite), a widely used method of quantum optimal control [BrumerShapiro2003, BrifNJP2010, SolaAAMOP2018](@cite). Its purpose is to find "controls" that steer a quantum system in a particular way. This is a prerequisite for next-generation quantum technology [DowlingPTRSA2003](@cite), such as quantum computing [NielsenChuang2000](@cite) or quantum sensing [DegenRMP2017](@cite). For example, in quantum computing with superconducting circuits [KochPRA2007](@cite), the controls are microwave pulses injected into the circuit in order to realize logical operations on the quantum states of the system, e.g., Ref. [GoerzNPJQI2017](@cite).

The quantum state of a system can be described by a complex vector ``\vert \Psi(t) \rangle`` that evolves under a differential equation of the form

```math
\begin{equation}\label{eq:tdse}
i \hbar \frac{\partial \vert\Psi(t)\rangle}{\partial t} = \hat{H}(\epsilon(t)) \vert\Psi(t)\rangle\,,
\end{equation}
```

where ``\hbar`` is the [reduced Planck constant](https://en.wikipedia.org/wiki/Planck_constant) and ``\hat{H}`` is a matrix whose elements depend in some way on the control function ``\epsilon(t)``. We generally know the initial state of the system ``\vert\Psi(t=0)\rangle`` and want to find an ``\epsilon(t)`` that minimizes some functional ``J`` that depends on the states at some final time ``T``, as well as running costs on ``\vert\Psi(t)\rangle`` and values of ``\epsilon(t)`` at intermediate times.

The defining feature of the GRAPE method is that it considers ``\epsilon(t)`` as piecewise constant, i.e., as a vector of values ``\epsilon_n``, for the ``n``'th interval of the time grid. This allows solving Eq. \eqref{eq:tdse} analytically for each time interval, and deriving an expression for the gradient ``\partial J / \partial \epsilon_n`` of the optimization functional with respect to the values of the control field. It results in an efficient numerical scheme for evaluating the full gradient [GoerzQ2022; Figure 1(a)](@cite). The scheme extends to situations where the functional is evaluated on top of *multiple* propagated states ``\{\vert \Psi_k(t) \rangle\}`` with an index ``k``, and multiple controls ``\epsilon_l(t)``, resulting in a vector of values ``\epsilon_{nl}`` with a double-index ``nl``. Once the gradient has been evaluated, in the original formulation of GRAPE [KhanejaJMR2005](@cite), the values ``\epsilon_{nl}`` would then be updated by taking a step with a fixed step width ``\alpha`` in the direction of the negative gradient, to iteratively minimize the value of the optimization functional ``J``. In practice, the gradient can also be fed into an arbitrary gradient-based optimizer, and in particular a quasi-Newton method like L-BFGS-B [ZhuATMS1997, LBFGSB.jl](@cite). This results in a dramatic improvement in stability and convergence [FouquieresJMR2011](@cite), and is assumed as the default in `GRAPE.jl`. Gradients of the time evolution operator can be evaluated to machine precision following [GoodwinJCP2015](@citet). The GRAPE method could also be extended to a true Hessian of the optimization functional [GoodwinJCP2016](@cite), which would be in scope for future versions of `GRAPE.jl`.



## Contents

```@contents
Depth = 2
Pages = [pair[2] for pair in Main.PAGES[2:end-1]]
```


## History

See the [Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases) on Github.
