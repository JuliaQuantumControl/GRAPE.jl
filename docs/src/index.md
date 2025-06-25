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

The `GRAPE.jl` package implements Gradient Ascent Pulse Engineering [KhanejaJMR2005](@cite), a widely used method of [quantum optimal control](@extref QuantumControl :doc:`index`). The quantum state of a system can be described by a complex vector ``\ket{\Psi(t)}`` that evolves under a differential equation of the form

```math
\def\ii{\mathrm{i}}
\begin{equation}\label{eq:tdse}
\ii \hbar \frac{\partial \ket{\Psi(t)}}{\partial t} = \hat{H}(\epsilon(t)) \ket{\Psi(t)}\,,
\end{equation}
```

where ``\hbar`` is the [reduced Planck constant](https://en.wikipedia.org/wiki/Planck_constant) and ``\hat{H}`` is a matrix whose elements depend in some way on the control function ``\epsilon(t)``. We generally know the initial state of the system ``\ket{\Psi(t=0)}`` and want to find an ``\epsilon(t)`` that minimizes some functional ``J`` that depends on the states at some final time ``T``, as well as running costs on ``\ket{\Psi(t)}`` and values of ``\epsilon(t)`` at intermediate times.

The defining feature of the GRAPE method is that it considers ``\epsilon(t)`` as piecewise constant, i.e., as a vector of pulse values ``\epsilon_n``, for the ``n``'th interval of the time grid. This allows solving Eq. \eqref{eq:tdse} analytically for each time interval, and deriving an expression for the gradient ``\partial J / \partial \epsilon_n`` of the optimization functional with respect to the values of the control field. The pulse values are then updated based on the gradient, in an efficient scheme detailed in [Background](@ref).


## Statement of Need


## Related Software


## Contents

```@contents
Depth = 2
Pages = [pair[2] for pair in Main.PAGES[2:end-1]]
```


## History

See the [Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases) on Github.
