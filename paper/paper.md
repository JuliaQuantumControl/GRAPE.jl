---
title: 'GRAPE.jl: Gradient Ascent Pulse Engineering in Julia'
tags:
  - Julia
  - quantum control
  - optimal control theory
  - GRAPE
authors:
  - name: Michael H. Goerz
    orcid: 0000-0003-2839-9976
    affiliation: "1"
  - name: Sebastián C. Carrasco
    orcid: 0000-0002-6512-9695
    affiliation: "1"
  - name: Vladimir S. Malinovsky
    orcid: 0000-0002-0243-9282
    affiliation: "1"
affiliations:
 - name: DEVCOM Army Research Laboratory, 2800 Powder Mill Road, Adelphi, MD 20783, United States
   index: 1
date: 16 June 2025
bibliography: paper.bib
---

# Summary

The `GRAPE.jl` package implements Gradient Ascent Pulse Engineering [@KhanejaJMR2005], a widely use method of quantum optimal control [@BrumerShapiro2003;@BrifNJP2010;@SolaAAMOP2018]. Its purpose is to find "controls" that steer a quantum system in a particular way. This is a prerequisite of next-generation quantum technology [@DowlingPTRSA2003] such as quantum computing [@NielsenChuang2000] or quantum sensing [@DegenRMP2017]. For example, in a quantum computing superconducting circuits [@KochPRA2007], the controls are microwave pulses injected into the circuit in order to realize logical operations on the quantum states of the system [e.g., @GoerzNPJQI2017].

The quantum state of a system can be described by a complex vector $\vert \Psi(t) \rangle$ that evolves under a differential equation of the form

\begin{equation}\label{eq:tdse}
i \hbar \frac{\partial \vert\Psi(t)\rangle}{\partial t} = \hat{H}(\epsilon(t)) \vert\Psi(t)\rangle\,,
\end{equation}

where $\hbar$ is the [reduced Planck constant](https://en.wikipedia.org/wiki/Planck_constant) and $\hat{H}$ is a matrix whose elements depend in some way on the control function $\epsilon(t)$. We generally know the initial state of the system $\vert\Psi(t=0)\rangle$ and want to find an $\epsilon(t)$ that minimizes some functional $J$ that depends on the states at some final time $T$, as well as running cost on $\vert\Psi(t)\rangle$ and values of $\epsilon(t)$ at intermediate time.

The defining feature of the GRAPE method is that it considers $\epsilon(t)$ as piecewise constant, i.e., as a vector of values $\epsilon_n$, for the $n$'th interval of the time grid. This allows to solve \autoref{eq:tdse} analytically for each time interval, and to derive an expression for the gradient $\partial J / \partial \epsilon_n$ of the optimization functional with respect to the values of the control field. It results in an efficient numerical scheme for evaluating the full gradient vector [@GoerzQ2022, {Figure 1(a)}]. The scheme extends to situations where there the functional is evaluated on top of *multiple* propagated states $\{\vert \Psi_k(t) \rangle\}$ with an index $k$, and multiple controls $\epsilon_l(t)$, resulting in a vector of values $\epsilon_{nl}$ with a double-index $nl$. Once the gradient has been evaluated, in the original formulation of GRAPE [@KhanejaJMR2005], the values $\epsilon_{nl}$ would then be updated by taking a step with a fixed step width $\alpha$ in the direction of the negative gradient, to iteratively minimize the value of the optimization functional $J$. In practice, the gradient can also be fed into an arbitrary gradient-based optimizer, and in particular a quasi-Newton method like L-BFGS-B [@ZhuATMS1997; @LBFGSB.jl]. This results in a dramatic improvement in stability and convergence, and is assumed as the default in `GRAPE.jl`. The GRAPE method could also be extended to a true Hessian of the optimization functional [@GoodwinJCP2016], which would be in scope for future versions of `GRAPE.jl`.


# Statement of need

A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. A description of how this software compares to other commonly-used packages in this research area

* Benefits of the Julia language (performance, parallelizability, composability)
* Flexibility: QuantumControl.jl framework [@QuantumControl.jl]
* Multiple states, multiple fields. Ensemble optimization.
* Non-linear Hamiltonians
* Machine precision gradients [@GoodwinJCP2015]
* quasi-Newton [@FouquieresJMR2011]
* Semi-automatic differentiation for arbitrary functionals [@GoerzQ2022]
* No fixed automatic differentiation framework
* Support for open quantum systems [@GoerzNJP2014]

Existing packages:

* SIMPSON [@TosnerJMR2009] (C)
* Spinach [@HogbenJMR2011] (Matlab)
* QuTIP [@JohanssonCPC2013] (Python)
* C3 [@WittlerPRA2021] (Python)
* QuOCS [@RossignoloCPC2023] (Python)
* QuanEstimation [@ZhangPRR2022] (Julia)
* pulse-finder [@pulse-finder] (Matlab)
* @QDYN (Fortran)

Full-AD [@LeungPRA2017; @quantum-optimal-control; @AbdelhafezPRA2019; @AbdelhafezPRA2020]

# Acknowledgements

Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Numbers
W911NF-23-2-0128 (MG) and W911NF-24-2-0044 (SC). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

# References
