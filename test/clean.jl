"""
    clean([distclean=false])

Clean up build/doc/testing artifacts. Restore to clean checkout state
(distclean)
"""
function clean(; distclean=false, _exit=true)

    _exists(name) = isfile(name) || isdir(name)
    _push!(lst, name) = _exists(name) && push!(lst, name)

    function _glob(folder, ending)
        if !_exists(folder)
            return []
        end
        [name for name in readdir(folder; join=true) if (name |> endswith(ending))]
    end

    function _glob_star(folder; except=[])
        if !_exists(folder)
            return []
        end
        [
            joinpath(folder, name) for
            name in readdir(folder) if !(name |> startswith(".") || name ∈ except)
        ]
    end


    ROOT = dirname(@__DIR__)

    ###########################################################################
    CLEAN = String[]
    _push!(CLEAN, joinpath(ROOT, "coverage"))
    _push!(CLEAN, joinpath(ROOT, "docs", "build"))
    _push!(CLEAN, joinpath(ROOT, "docs", "LocalPreferences.toml"))
    _push!(CLEAN, joinpath(ROOT, "test", "LocalPreferences.toml"))
    append!(CLEAN, _glob(ROOT, ".info"))
    append!(CLEAN, _glob(joinpath(ROOT, ".coverage"), ".info"))
    ###########################################################################

    ###########################################################################
    DISTCLEAN = String[]
    for folder in ["", "docs", "test"]
        _push!(DISTCLEAN, joinpath(joinpath(ROOT, folder), "Manifest.toml"))
    end
    _push!(DISTCLEAN, joinpath(ROOT, "docs", "Project.toml"))
    append!(DISTCLEAN, _glob_star(joinpath(ROOT, "test", "data")))
    append!(DISTCLEAN, _glob_star(joinpath(ROOT, "docs", "data")))
    _push!(DISTCLEAN, joinpath(ROOT, ".JuliaFormatter.toml"))
    ###########################################################################

    for name in CLEAN
        @info "rm $name"
        rm(name, force=true, recursive=true)
    end
    if distclean
        for name in DISTCLEAN
            @info "rm $name"
            rm(name, force=true, recursive=true)
        end
        if _exit
            @info "Exiting"
            exit(0)
        end
    end

end

distclean() = clean(distclean=true)
