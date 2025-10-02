# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumControl: load_optimization

@testset "README" begin

    temp_dir = mktempdir()
    readme_path = joinpath(@__DIR__, "..", "README.md")
    readme_example_path = joinpath(temp_dir, "readme_example.jl")

    open(readme_path, "r") do readme_file
        open(readme_example_path, "w") do example_file
            in_julia_block = false
            for line in eachline(readme_file)
                if line == "```julia"
                    in_julia_block = true
                elseif line == "```" && in_julia_block
                    in_julia_block = false
                elseif in_julia_block
                    println(example_file, line)
                end
            end
        end
    end

    cd(temp_dir) do
        # `cd` make sure `tls_opt.jld2` is created in `temp_dir`
        include(readme_example_path)
    end
    tls_opt_jld2 = joinpath(temp_dir, "tls_opt.jld2")
    @test isfile(tls_opt_jld2)
    if isfile(tls_opt_jld2)
        result = load_optimization(tls_opt_jld2)
        @test result.converged
        @test result.J_T < 1e-3
    end

end
