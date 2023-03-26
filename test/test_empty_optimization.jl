using Test
using StableRNGs
using QuantumControl: hamiltonian, optimize, ControlProblem, Objective
using QuantumControl.Controls: get_controls
using QuantumControl.Functionals: J_T_re
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector

@testset "empty optimization" begin

    # Test that trying to run an optimization without any controls produces a
    # meaningful error message

    rng = StableRNG(2264511904)

    N = 10
    H = random_matrix(N; rng)
    objectives = [
        Objective(;
            initial_state=random_state_vector(N; rng),
            generator=H,
            target_state=random_state_vector(N; rng)
        )
    ]

    @test length(get_controls(objectives)) == 0

    tlist = collect(range(0; length=1001, step=1.0))

    problem = ControlProblem(; objectives, tlist, J_T=J_T_re)

    msg = "no controls in objectives: cannot optimize"
    @test_throws ErrorException(msg) optimize(problem; method=:GRAPE)

end
