.PHONY: help clean paper

DOCKER ?= docker
FOLDER ?= $(PWD)

define PRINT_HELP_JLSCRIPT
rx = r"^([a-z0-9A-Z_-]+):.*?##[ ]+(.*)$$"
for line in eachline()
    m = match(rx, line)
    if !isnothing(m)
        target, help = m.captures
        println("$$(rpad(target, 20)) $$help")
    end
end
println("""

Compiling the manuscript requires Docker or a compatible containerization system.

See `README.md` for details.
""")
endef
export PRINT_HELP_JLSCRIPT


help:  ## Show this help
	@julia -e "$$PRINT_HELP_JLSCRIPT" < $(MAKEFILE_LIST)


clean: ## Clean up generated files
	rm -f paper.pdf
	rm -rf jats

paper: paper.pdf ## Compile the JOSS manuscript paper.md

paper.pdf: paper.md paper.bib
	$(DOCKER) run --rm --volume $(FOLDER):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara -o pdf -p paper.md
