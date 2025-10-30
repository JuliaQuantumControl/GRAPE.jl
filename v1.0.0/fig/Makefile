# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: CC0-1.0

PDFLATEXOPTS = -file-line-error -interaction=nonstopmode -halt-on-error
ALL = grape_scheme.pdf grape_scheme.png

all: grape_scheme.png

%.pdf: %.tex
	@echo "\n** generate figure: $< -> $@"
	@pdflatex ${PDFLATEXOPTS} $<

grape_scheme.png: grape_scheme.pdf
	convert -background white -alpha remove -density 600 -resize 2142x1071 -extent 2142x1071 -gravity center $< $@

clean:
	@rm -f *.aux
	@rm -f *.log
	@rm -f *.toc
	@rm -f *.blg
	@rm -rf *.out
	@rm -f *.bak
	@rm -f *.ilg
	@rm -f *.snm
	@rm -f *.nav
	@rm -f *.table
	@rm -f *.dvi
	@rm -f *.fls
	@rm -f *.bbl
	@rm -f *~
	@rm -f *Notes.bib
	@rm -f *-eps-converted-to.pdf
	@rm -f *.fdb_latexmk
	@rm -f *.synctex.gz*
	@rm -f ./latexrun
	@rm -f grape_scheme.pdf

distclean: clean
	rm -f $(ALL)

.PHONY: all clean distclean
