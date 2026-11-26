<!--
SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: CC-BY-4.0
-->

# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For releases pre-1.0, see the [GitHub Releases](https://github.com/JuliaQuantumControl/GRAPE.jl/releases).


## [Unreleased]

* Changed: The minimum supported version of QuantumControl is now 0.11.7 [[#128]]
* Fixed: With `rethrow_exceptions=true`, an exception during the optimization left the hook registered that saves the result when Julia exits during an optimization with `QuantumControl.@optimize_or_load`. At the exit of the Julia process, that hook could overwrite a later result in the same file with the partial result of the failed optimization. Also, if another `atexit` hook was registered during the optimization, the wrong hook was removed at the end of the optimization. As a result of the fix, a script with `rethrow_exceptions=true` no longer writes the partial result (with the message "Abort: ATEXIT") to the file of `QuantumControl.@optimize_or_load` when an exception ends the optimization. An interruption of the Julia process by a signal (e.g., SIGINT or SIGTERM) still saves the partial result [[#128]]


## [v1.3.0] — 2026-11-23

* Changed: The minimum supported version of QuantumGradientGenerators is now 0.2 [[#124]]
* Changed: For an explicit `optimizer = LBFGSB.L_BFGS_B(nmax, mmax)`, the number of limited memory corrections `lbfgsb_m` now defaults to `mmax` instead of 10 [[#122]]
* Changed: The `:lower_bounds` and `:upper_bounds` in `pulse_options` must be a single number for all time intervals, or a vector with one value for each time interval. A vector of any other length, including a single-element vector, throws an `ArgumentError` [[#123]]
* Changed: The overlaps with the target states from the most recent evaluation of the functional are now in the workspace (`wrk.tau_vals`). `result.tau_vals` is only updated at the end of each iteration [[#127]]
* Changed: Until the guess has been evaluated, `result.J_T` and `result.tau_vals` are `NaN` instead of zero, so that an optimization that ends before the guess has been evaluated (e.g., with Ctrl-C) does not report a value of zero for the functional [[#127]]
* Added: Documentation for the keyword arguments `lbfgsb_m`, `lbfgsb_factr`, `lbfgsb_pgtol`, and `lbfgsb_iprint` [[#122]]
* Added: Documentation for the keyword arguments `iter_start`, `iter_stop`, `continue_from`, and `use_threads`, and for the iteration number that the `callback` receives for the guess [[#124]]
* Fixed: The `lbfgsb_m` keyword argument for the number of limited memory corrections in the L-BFGS-B optimizer was ignored when allocating the optimizer, which always had room for 10 corrections. For `lbfgsb_m > 10`, this corrupted memory, resulting in crashes. A user-supplied L-BFGS-B `optimizer` that was allocated for fewer corrections than `lbfgsb_m` or for the wrong number of control values now throws an `ArgumentError` [[#120]]
* Fixed: For more than one control, the per-control `:lower_bounds` and `:upper_bounds` in `pulse_options` were applied to the wrong time intervals, interleaved across all controls [[#121]]
* Fixed: Values of `Inf` in the `:upper_bounds` and `-Inf` in the `:lower_bounds` of `pulse_options` now fall back to the global `upper_bound` and `lower_bound`, as documented. Previously, they removed the global bound for that time interval [[#123]]
* Fixed: Without an upper bound, the L-BFGS-B optimizer treated every control value as bounded from above by `Inf`, so it always solved a constrained problem, and a `lower_bound` alone was passed as a two-sided bound. Optimizations without bounds now take a different path through L-BFGS-B, so their results differ from previous versions. In particular, the first iteration is no longer a short steepest-descent step [[#123]]
* Fixed: Invalid bounds now throw an `ArgumentError` that identifies the control and time interval: a `NaN` bound, an upper bound of `-Inf`, a lower bound of `Inf`, or a lower bound that exceeds the upper bound. Previously, such bounds were silently ignored, or the optimization ended before the first iteration with the message "ERROR: NO FEASIBLE SOLUTION" [[#123]]
* Fixed: The propagators for GRAPE were not required to be piecewise, so that a propagation method that is not piecewise (e.g., an ODE solver) was accepted. All propagations now use `piecewise=true`, and `init_prop` throws an error for a method that is not piecewise. The `piecewise`, `backward`, and `parameters` arguments that GRAPE passes to the propagators can no longer be overridden by keyword arguments [[#124]]
* Fixed: The `prop_callback` of the backward propagation of the gradient generator received the `observables` of the forward propagation instead of its own (`grad_prop_observables`) [[#124]]
* Fixed: An explicit `callback=nothing` (e.g., stored in a `ControlProblem`) ended the optimization with an exception. It is now treated as no callback [[#124]]
* Fixed: An `iter_stop` that is not larger than the starting iteration (`iter_start`, or the last iteration of `continue_from`) is now respected: only the guess is evaluated. Previously, L-BFGS-B ran one more iteration, and an Optim.jl optimizer ran up to `iter_stop - iter_start` iterations, ignoring the iterations of `continue_from`. A `callback` that sets `result.converged` for the guess now also stops an L-BFGS-B optimization before the first iteration [[#124]]
* Fixed: If L-BFGS-B projects a guess that violates the bounds into the bounds, the workspace now uses the projected guess, with a warning. Previously, `pulsevals_guess`, the gradient for the guess, and the pulse update in the first iteration were inconsistent with the projected guess [[#124]]
* Fixed: `result.J_a`, and the `J_a` and `J` columns printed with `print_iters`, were wrong for a negative running cost `J_a` or for `lambda_a=0` [[#124]]
* Fixed: A singular `:upper_bound` or `:lower_bound` key in `pulse_options` was silently ignored. It now throws an `ArgumentError` pointing to `:upper_bounds` and `:lower_bounds` [[#124]]
* Fixed: The `result.message` from L-BFGS-B no longer has trailing whitespace [[#124]]
* Fixed: The documented default labels of `print_iter_info` were wrong, and the documentation of `taylor_grad_max_order` referred to the wrong value of `taylor_grad_check_convergence` [[#124]]
* Fixed: The documentation of the propagation keyword arguments now describes their actual precedence: any keyword argument (e.g., a global `prop_method`) overrides the properties of all trajectories, including properties with a more specific prefix like `fw_prop_method`. The documentation previously stated that `fw_prop_` and `bw_prop_` settings override any `prop_` setting [[#124]]
* Fixed: For states that support in-place operations (e.g., `Vector`), `result.states`, and the state of the forward propagators as seen by a `callback`, were the initial states instead of the states at final time for the current pulses [[#126]]
* Fixed: `result.optimized_controls` was only set at the end of the optimization, so a `callback` or `check_convergence` saw the guess controls together with the `states` and `J_T` of the current iteration, and continuing from a result saved on an unexpected exit (by `QuantumControl.@optimize_or_load`) restarted from the guess controls. It is now updated in every iteration [[#127]]
* Fixed: If an optimization was interrupted by an exception (e.g., Ctrl-C), `result.optimized_controls` and `result.tau_vals` could be for an incomplete iteration (e.g., a point in a line search), unlike the other attributes of the result. The result now consistently describes the last completed iteration [[#127]]
* Fixed: Continuing an optimization (`continue_from`) from the result of a different method (e.g., a `KrotovResult`) modified that result [[#127]]


## [v1.2.0] — 2026-09-16

* Changed: The minimum supported Julia version is now 1.10 (LTS)
* Changed: The minimum supported versions of dependencies are now QuantumControl 0.11.5 and QuantumGradientGenerators 0.1.9. GRAPE does not work with QuantumGradientGenerators 0.1.8, which lacks the type-based `supports_inplace` trait of QuantumPropagators 0.9
* Changed: An `optimizer` from Optim.jl now requires Optim 2. Support for Optim 1 is dropped. Only first-order optimizers (e.g., `Optim.LBFGS()`) are accepted [[#114], replacing [#110]]
* Changed: The convergence tolerances of an Optim.jl `optimizer` use the keyword arguments `x_abstol`, `x_reltol`, `f_abstol`, `f_reltol`, and `g_abstol`, like `Optim.Options`. The previous `x_tol`, `f_tol`, and `g_tol` remain as aliases for `x_abstol`, `f_reltol`, and `g_abstol`
* Fixed: With an Optim.jl `optimizer`, the `callback` for the guess received iteration number 1 instead of 0, which shifted all iteration numbers by one and ran one iteration fewer than `iter_stop`
* Fixed: With an Optim.jl `optimizer`, `search_direction` and `step_width` for `Optim.ConjugateGradient` returned the search direction of the following iteration
* Fixed: A `callback` that mutates `pulsevals` for an Optim.jl `optimizer` now throws an error instead of silently desynchronizing the workspace from the optimizer


## [v1.1.0] — 2026-06-20

* Added: Support for state-dependent running costs [[#53], [#103], followup in [#105]]
* Fixed: Bug in `gradient_method = :taylor` incorrectly accessing `pulsevals`, leading to incorrect gradients when the control problem contains more than one control [fixed as part of [#103]]


## [v1.0.0] — 2025-10-30

Initial stable release. No breaking changes compared to [v0.8.1].

[Unreleased]: https://github.com/JuliaQuantumControl/GRAPE.jl/compare/v1.3.0..HEAD
[v1.3.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.3.0
[v1.2.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.2.0
[v1.1.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.1.0
[v1.0.0]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v1.0.0
[v0.8.1]: https://github.com/JuliaQuantumControl/GRAPE.jl/releases/tag/v0.8.1
[#128]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/128
[#127]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/127
[#126]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/126
[#124]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/124
[#123]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/123
[#122]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/122
[#121]: https://github.com/JuliaQuantumControl/GRAPE.jl/issues/121
[#120]: https://github.com/JuliaQuantumControl/GRAPE.jl/issues/120
[#114]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/114
[#110]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/110
[#105]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/105
[#103]: https://github.com/JuliaQuantumControl/GRAPE.jl/pull/103
[#53]: https://github.com/JuliaQuantumControl/GRAPE.jl/issues/53
