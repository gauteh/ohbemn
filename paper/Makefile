# Directory containing source (Markdown) files
source := .

# Directory containing pdf files
output := .

# All markdown files in src/ are considered sources
sources := $(wildcard $(source)/main.tex)

# Convert the list of source files (Markdown files in directory src/)
# into a list of output files (PDFs in directory print/).
objects := $(patsubst %.tex,%.pdf,$(subst $(source),$(output),$(sources)))

.PHONY: clean

all: $(objects)

# Recipe for converting a Markdown file into PDF using Pandoc
$(output)/%.pdf: $(source)/%.tex
	latexmk -f -pdf -interaction=nonstopmode $<

# SVGFILES := $(wildcard *.svg)
# pngs: $(SVGFILES:%.svg=%.pdf)
# %.png : %.svg
#         inkscape $*.svg --export-dpi=250 -o $*.png

dist: all
	mkdir -p dist
	python distfig.py main.tex ./dist/
	cp ametsoc* dist/
	cp *.bib dist/
	cp Makefile dist/

clean:
	latexmk -c
	rm -f *.bbl
	rm -f *.run.xml
	rm -f *.tdo
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f texput.fls
	rm -f tfcad.bst
	rm -f *.synctex.gz
	rm -f $(output)/main.pdf


