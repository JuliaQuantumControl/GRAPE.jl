# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: CC0-1.0

.PHONY: help clean paper preprint pandoc

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
	rm -f paper.preprint.*
	rm -rf jats

paper: paper.pdf ## Compile the JOSS manuscript paper.md

pandoc:  ## Compile the manuscript directly with `pandoc` (light-weight preview)
	pandoc --citeproc --csl=apa.csl -o paper.pdf --pdf-engine=lualatex paper.md

paper.pdf: paper.md paper.bib
	$(DOCKER) run --rm --volume $(FOLDER):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara -o pdf -p paper.md

preprint: paper.preprint.pdf  ## Compile the manuscript in "preprint" format

paper.preprint.tex: paper.md paper.bib
	$(DOCKER) run --rm --volume $(FOLDER):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara -o preprint -p paper.md

paper.preprint.pdf: paper.preprint.tex
	pdflatex $<
	pdflatex $<
