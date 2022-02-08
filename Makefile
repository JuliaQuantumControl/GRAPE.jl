.PHONY: help test docs clean distclean devrepl codestyle
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-z0-9A-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s %s" % (target, help))
print("""
Instead of "make test", consider "make devrepl" if you want to run the test
suite or generate the docs repeatedly.

Make sure you have Revise.jl installed in your standard Julia environment
""")
endef
export PRINT_HELP_PYSCRIPT

GITORIGIN := $(shell git config remote.origin.url)

help:  ## show this help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# We want to test against checkouts of QuantumControl packages
QUANTUMCONTROLBASE ?= ../QuantumControlBase.jl
QUANTUMPROPAGATORS ?= ../QuantumPropagators.jl
KROTOV ?= ../Krotov.jl
QUANTUMCONTROL ?= ../QuantumControl.jl
GRAPELINESEARCHANALYSIS ?= ../GRAPELinesearchAnalysis.jl


define DEV_PACKAGES
using Pkg;
Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaQuantumControl/QuantumControlRegistry.git"))
Pkg.develop(path="$(QUANTUMCONTROLBASE)");
Pkg.develop(path="$(QUANTUMPROPAGATORS)");
endef
export DEV_PACKAGES

define ENV_PACKAGES
$(DEV_PACKAGES)
Pkg.develop(path="$(KROTOV)");
Pkg.develop(path="$(QUANTUMCONTROL)");
Pkg.develop(path="$(GRAPELINESEARCHANALYSIS)");
Pkg.develop(PackageSpec(path=pwd()));
Pkg.instantiate()
endef
export ENV_PACKAGES


JULIA ?= julia
DEV_PROJECT_TOMLS = $(QUANTUMCONTROLBASE)/Project.toml $(QUANTUMPROPAGATORS)/Project.toml $(QUANTUMCONTROL)/Project.toml

Manifest.toml: Project.toml $(DEV_PROJECT_TOMLS)
	$(JULIA) --project=. -e "$$DEV_PACKAGES;Pkg.instantiate()"


test:  test/Manifest.toml ## Run the test suite
	$(JULIA) --project=test --color=auto --startup-file=yes --code-coverage="user" --depwarn="yes" --check-bounds="yes" -e 'include("test/runtests.jl")'
	@echo "Done. Consider using 'make devrepl'"


test/Manifest.toml: test/Project.toml $(DEV_PROJECT_TOMLS)
	$(JULIA) --project=test -e "$$ENV_PACKAGES"


docs/Manifest.toml: test/Manifest.toml
	cp test/*.toml docs/

devrepl: test/Manifest.toml docs/Manifest.toml  ## Start an interactive REPL for testing and building documentation
	@$(JULIA) --threads auto --project=test --banner=no --startup-file=yes -e 'include("test/init.jl")' -i



docs: docs/Manifest.toml  ## Build the documentation
	$(JULIA) --project=test docs/make.jl
	@echo "Done. Consider using 'make devrepl'"


clean: ## Clean up build/doc/testing artifacts
	rm -f src/*.cov test/*.cov examples/*.cov
	rm -f test/examples/*.*
	for file in examples/*.jl; do rm -f docs/src/"$${file%.jl}".*; done
	rm -rf docs/build


codestyle: test/Manifest.toml ../.JuliaFormatter.toml ## Apply the codestyle to the entire project
	$(JULIA) --project=test -e 'using JuliaFormatter; format(".", verbose=true)'
	@echo "Done. Consider using 'make devrepl'"


distclean: clean ## Restore to a clean checkout state
	rm -f Manifest.toml test/Manifest.toml
	rm -f docs/Manifest docs/Project.toml
	rm -f test/data/*.jld2 docs/data/*.jld2
