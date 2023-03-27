# generate examples
import Literate

println("Start generating Literate.jl examples")

EXAMPLEDIR = joinpath(@__DIR__, "..", "examples")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples")
mkpath(GENERATEDDIR)
for example in readdir(EXAMPLEDIR)
    if endswith(example, ".jl")
        input = abspath(joinpath(EXAMPLEDIR, example))
        Literate.markdown(input, GENERATEDDIR)
        Literate.notebook(input, GENERATEDDIR, execute=("CI" ∈ keys(ENV)))
    elseif any(endswith.(example, [".png", ".jpg", ".gif"]))
        cp(joinpath(EXAMPLEDIR, example), joinpath(GENERATEDDIR, example); force=true)
    else
        @warn "ignoring $example"
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATEDDIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
    foreach(file -> endswith(file, ".pvd") && rm(file), readdir())
end

println("Finished generating Literate.jl examples")
