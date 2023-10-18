import Downloads

const PROJECTDIR = dirname(Base.active_project())
projectdir(names...) = joinpath(PROJECTDIR, names...)
datadir(names...) = projectdir("data", names...)

DOWNLOADS = Dict(
    "https://raw.githubusercontent.com/JuliaQuantumControl/GRAPE.jl/data-dump/TLS/opt_result_LBFGSB.jld2" =>
        datadir("TLS", "GRAPE_opt_result_LBFGSB.jld2"),
    "https://raw.githubusercontent.com/JuliaQuantumControl/GRAPE.jl/data-dump/TLS/opt_result_OptimLBFGS.jld2" =>
        datadir("TLS", "GRAPE_opt_result_OptimLBFGS.jld2"),
    "https://raw.githubusercontent.com/JuliaQuantumControl/GRAPE.jl/data-dump/GATE_OCT.jld2" =>
        datadir("GRAPE_GATE_OCT.jld2"),
    "https://raw.githubusercontent.com/JuliaQuantumControl/GRAPE.jl/data-dump/PE_OCT.jld2" =>
        datadir("GRAPE_PE_OCT.jld2"),
    "https://raw.githubusercontent.com/JuliaQuantumControl/GRAPE.jl/data-dump/PE_OCT_direct.jld2" =>
        datadir("GRAPE_PE_OCT_direct.jld2"),
)

function download_dump(url, destination; force=false, verbose=true)
    if !isfile(destination) || force
        verbose && (@info "Downloading $url => $destination")
        mkpath(dirname(destination))
        Downloads.download(url, destination)
    else
        verbose && (@info "$destination OK")
    end
end

@info "Download Dumps"
for (url, destination) in DOWNLOADS
    download_dump(url, destination)
end
