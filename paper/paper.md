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
  - name: Alastair Marshall
    orcid: 0000-0001-7772-2173
    affiliation: "2"
  - name: Vladimir S. Malinovsky
    orcid: 0000-0002-0243-9282
    affiliation: "1"
affiliations:
 - name: DEVCOM Army Research Laboratory, 2800 Powder Mill Road, Adelphi, MD 20783, United States
   index: 1
 - name: Universität Ulm,  Institute for Quantum Optics, Albert-Einstein-Allee 11, 89081 Ulm, Germany
   index: 2
date: 15 July 2025
bibliography: paper.bib
---

# Summary

[The `GRAPE.jl` package](https://github.com/JuliaQuantumControl/GRAPE.jl) implements Gradient Ascent Pulse Engineering [@KhanejaJMR2005], a widely used method of quantum optimal control [@BrumerShapiro2003; @BrifNJP2010; @SolaAAMOP2018]. Its purpose is to find "controls" that steer a quantum system in a particular way. This is a prerequisite for next-generation quantum technology [@DowlingPTRSA2003], such as quantum computing [@NielsenChuang2000] or quantum sensing [@DegenRMP2017]. For example, in quantum computing with superconducting circuits [@KochPRA2007], the controls are microwave pulses injected into the circuit in order to realize logical operations on the quantum states of the system [e.g., @GoerzNPJQI2017].

The quantum state of a system can be described numerically by a complex vector $\vert \Psi(t) \rangle$ that evolves under a differential equation of the form

\begin{equation}\label{eq:tdse}
i \hbar \frac{\partial \vert\Psi(t)\rangle}{\partial t} = \hat{H}(\epsilon(t)) \vert\Psi(t)\rangle\,,
\end{equation}

where $\hbar$ is the [reduced Planck constant](https://en.wikipedia.org/wiki/Planck_constant) and $\hat{H}$ is a matrix whose elements depend in some way on the control function $\epsilon(t)$. We generally know the initial state of the system $\vert\Psi(t=0)\rangle$ and want to find an $\epsilon(t)$ that minimizes some real-valued functional $J$ that depends on the states at some final time $T$, as well as running costs on $\vert\Psi(t)\rangle$ and values of $\epsilon(t)$ at intermediate times. A common example with be the square-modulus of the overlap with a target state.

The defining feature of the GRAPE method is that it considers $\epsilon(t)$ as piecewise constant, i.e., as a vector of values $\epsilon_n$, for the $n$'th interval of the time grid. This allows solving \autoref{eq:tdse} for each time interval, and deriving an expression for the gradient $\partial J / \partial \epsilon_n$ of the optimization functional with respect to the values of the control field. It results in an efficient numerical scheme for evaluating the full gradient [@GoerzQ2022, {Figure 1(a)}]. The scheme extends to situations where the functional is evaluated on top of *multiple* propagated states $\{\vert \Psi_k(t) \rangle\}$ with an index $k$, and multiple controls $\epsilon_l(t)$, resulting in a vector of values $\epsilon_{nl}$ with a double-index $nl$. Once the gradient has been evaluated, in the original formulation of GRAPE [@KhanejaJMR2005], the values $\epsilon_{nl}$ would then be updated by taking a step with a fixed step width $\alpha$ in the direction of the negative gradient, to iteratively minimize the value of the optimization functional $J$. In practice, the gradient can also be fed into an arbitrary gradient-based optimizer, and in particular a quasi-Newton method like L-BFGS-B [@ZhuATMS1997; @LBFGSB.jl]. This results in a dramatic improvement in stability and convergence [@FouquieresJMR2011], and is assumed as the default in `GRAPE.jl`. Gradients of the time evolution operator can be evaluated to machine precision following @GoodwinJCP2015. The GRAPE method could also be extended to a true Hessian of the optimization functional [@GoodwinJCP2016], which would be in scope for future versions of `GRAPE.jl`.

# Statement of Need

There have been a number of implementations of the GRAPE method in different contexts. GRAPE was originally developed and adopted in the NMR community, e.g., as part of `SIMPSON` [@TosnerJMR2009] in C, and later as part of `Spinach` [@HogbenJMR2011] and `pulse-finder` [@pulse-finder] in Matlab. More recent implementations in Python, geared towards more general purposes like quantum information, are found as part of the `QuTIP` library [@JohanssonCPC2013], `C3` [@WittlerPRA2021], `QuOCS` [@RossignoloCPC2023], and `QuanEstimation` [@ZhangPRR2022]. The implementation of `GRAPE.jl` is also inspired by earlier work in the `QDYN` library in Fortran [-@QDYN]. `GRAPE.jl` exploits the unique strengths of the Julia programming language [@BezansonSIREV2017] to avoid common shortcomings in existing implementations.

As a compiled language geared towards scientific computing, Julia delivers numerical performance similar to that of Fortran, while providing much greater flexibility due to the expressiveness of the language. The numerical cost of the GRAPE method is dominated by the cost of evaluating the time evolution of the quantum system. `GRAPE.jl` delegates this to efficient piecewise-constant propagators in `QuantumPropagators.jl` [@QuantumPropagators.jl] or the general-purpose `DifferentialEquations.jl` framework [@RackauckasJORS2017].

`GRAPE.jl` builds on the concepts defined in `QuantumControl.jl` [@QuantumControl.jl] to allow functionals that depend on an arbitrary set of "trajectories" $\{ \vert \Psi_k(t) \rangle\}$, each evolving under a potentially different $\hat{H}_k$. In contrast to the common restriction to a single state $\vert\Psi\rangle$ or a single unitary $\hat{U}$ as the dynamical state, this enables ensemble optimization for robustness against noise [e.g., @GoerzPRA2014]. The optimization over multiple trajectories is parallelized. This makes the optimization of quantum gates more efficient, by tracking the logical basis states instead of the gate $\hat{U}(t)$. Each $\hat{H}_k$ may depend on an arbitrary number of controls $\{\epsilon_l(t)\}$ in an arbitrary way, going beyond the common assumption of linear controls, $\hat{H} = \hat{H}_0 + \epsilon(t) \hat{H}_1$.

Julia's core feature of [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) allows the user to define custom, problem-specific data structures with performance-optimized linear algebra operations. This gives `GRAPE.jl` great flexibility to work with any custom data structures for quantum states $\vert \Psi_k(t) \rangle$ or the matrices $\hat{H}_k(\{\epsilon_l(t)\})$, and enables a wide range of applications, from NMR spin systems to superconducting circuits or trapped atoms in quantum computing, to systems with spatial degrees of freedom [e.g., @DashAVSQS2024]. This also includes open quantum systems, as the structure of \autoref{eq:tdse} holds not just for the standard Schrödinger equation, but also for the Liouville equation, where $\vert\Psi_k\rangle$ is replaced by a (vectorized) density matrix and $\hat{H}$ becomes a Liouvillian super-operator [@GoerzNJP2014].

The rise of machine learning generated considerable interest in using the capabilities of frameworks like Tensorflow [@Tensorflow], PyTorch [@PaszkeNIPS2019], or JAX [@JAX] for automatic differentiation (AD) [@Griewank2008] to evaluate the gradient of the optimization functional. This has the benefit that it allows for arbitrary functionals [@LeungPRA2017; @quantum-optimal-control; @AbdelhafezPRA2019; @AbdelhafezPRA2020]. In contrast, the GRAPE method and all of its existing implementations are formulated only for a "standard" set of functionals that essentially measure the overlap of a propagated state with a target state. Unfortunately, AD comes with a large numerical overhead that makes the method impractical. @GoerzQ2022 introduced the use of "semi-automatic differentiation" that limits the numerical cost to exactly that of the traditional GRAPE scheme. It does this by employing AD only for the evaluation of the derivative $\partial J/\partial \langle \Psi_k(T) \vert$, and only if that derivative cannot be evaluated analytically. `GRAPE.jl` is built on the resulting generalized GRAPE scheme. As necessary, it can use any available AD framework in the Julia ecosystem to enable the minimization of non-analytical functionals, such as entanglement measures [@GoerzPRA2015; @WattsPRA2015].


# Acknowledgements

Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Numbers
W911NF-23-2-0128 (MG) and W911NF-24-2-0044 (SC). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

# References
