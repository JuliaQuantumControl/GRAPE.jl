# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl: hamiltonian, optimize, ControlProblem, Trajectory
using QuantumControl.Functionals: J_T_sm
using GRAPE
using GRAPE: GrapeWrk

import GRAPE.LBFGSB

# The number of elements of the `wa` work array that the Fortran routine
# `setulb` requires for `n` variables and `m` limited memory corrections
wa_len(n, m) = (2m + 5) * n + 11m^2 + 8m


function tls_problem(; kwargs...)
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    tlist = collect(range(0, 5; length = 201))
    H = hamiltonian(σz, (σx, t -> 0.1))
    traj = Trajectory(ComplexF64[1, 0], H; target_state = ComplexF64[0, 1])
    return ControlProblem(
        [traj],
        tlist;
        J_T = J_T_sm,
        prop_method = :expprop,
        iter_stop = 5,
        kwargs...
    )
end


@testset "lbfgsb_m allocation" begin
    # https://github.com/JuliaQuantumControl/GRAPE.jl/issues/120
    n = 200
    for m in (10, 3, 100)
        wrk = GrapeWrk(tls_problem(; lbfgsb_m = m))
        @test wrk.optimizer isa LBFGSB.L_BFGS_B
        @test wrk.optimizer.mmax == m
        @test length(wrk.optimizer.wa) == wa_len(n, m)
    end
    wrk = GrapeWrk(tls_problem())
    @test wrk.optimizer.mmax == 10
end


@testset "lbfgsb_m optimization" begin
    res = optimize(tls_problem(; lbfgsb_m = 50); method = GRAPE, rethrow_exceptions = true)
    @test res.iter == 5
    @test res.J_T < res.J_T_prev
end


@testset "lbfgsb_m invalid" begin
    n = 200
    msg = "`lbfgsb_m` must be a positive integer, not 0"
    @test_throws ArgumentError(msg) GrapeWrk(tls_problem(; lbfgsb_m = 0))
    # optimizer allocated for too few corrections
    optimizer = LBFGSB.L_BFGS_B(n, 10)
    @test_throws ArgumentError GrapeWrk(tls_problem(; optimizer, lbfgsb_m = 20))
    # optimizer allocated for the wrong number of variables
    optimizer = LBFGSB.L_BFGS_B(n - 1, 10)
    @test_throws ArgumentError GrapeWrk(tls_problem(; optimizer))
    optimizer = LBFGSB.L_BFGS_B(n + 1, 10)
    @test_throws ArgumentError GrapeWrk(tls_problem(; optimizer))
    # The error message shows how to allocate a suitable optimizer
    exc = try
        GrapeWrk(tls_problem(; optimizer, lbfgsb_m = 20))
    catch e
        e
    end
    @test exc isa ArgumentError
    @test contains(exc.msg, "Use `LBFGSB.L_BFGS_B(200, 20)`")
end


@testset "lbfgsb_m custom optimizer" begin
    # `setulb` stores m² in `isave[2]`, which shows the `m` that L-BFGS-B
    # actually used
    n = 200
    # Without `lbfgsb_m`, the number of corrections is `mmax`
    optimizer = LBFGSB.L_BFGS_B(n, 30)
    problem = tls_problem(; optimizer)
    res = optimize(problem; method = GRAPE, rethrow_exceptions = true)
    @test res.iter == 5
    @test res.J_T < res.J_T_prev
    @test optimizer.isave[2] == 30^2
    # An optimizer for more corrections than `lbfgsb_m` is fine
    optimizer = LBFGSB.L_BFGS_B(n, 30)
    problem = tls_problem(; optimizer, lbfgsb_m = 20)
    res = optimize(problem; method = GRAPE, rethrow_exceptions = true)
    @test res.iter == 5
    @test res.J_T < res.J_T_prev
    @test optimizer.isave[2] == 20^2
    # Without an explicit optimizer, the default is `lbfgsb_m = 10`
    for (kwargs, m) in (((;), 10), ((; lbfgsb_m = 15), 15))
        workspaces = []
        callback(wrk, iter) = (iter == 0) && push!(workspaces, wrk)
        problem = tls_problem(; callback, kwargs...)
        res = optimize(problem; method = GRAPE, rethrow_exceptions = true)
        @test res.iter == 5
        @test workspaces[1].optimizer.mmax == m
        @test workspaces[1].optimizer.isave[2] == m^2
    end
end
