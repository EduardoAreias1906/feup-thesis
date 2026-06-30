# Makefile for FEUP Thesis

# Defaults
.PHONY: all report preparation dissertation feup_thesis thesis clean check_deps check_latex

# Default target — build the new FEUP template
all: feup_thesis

# Alias for preparation
report: preparation

# Build the dissertation preparation report
# Logic: Check for latexmk -> if yes, use it.
#        If no, check for biber -> if yes, use pdflatex + biber sequence.
#        If no biber, use pdflatex only (warn user).
preparation:
	@echo "Building Dissertation Preparation Report..."
	@cd dissertation-preparation && \
	if command -v latexmk >/dev/null 2>&1; then \
		echo "Found latexmk, using it..."; \
		latexmk -pdf -interaction=nonstopmode main.tex; \
	else \
		echo "latexmk not found. Falling back to manual build sequence."; \
		pdflatex -interaction=nonstopmode main.tex; \
		if command -v bibtex >/dev/null 2>&1; then \
			echo "Found bibtex, running bibliography..."; \
			bibtex main; \
			pdflatex -interaction=nonstopmode main.tex; \
			pdflatex -interaction=nonstopmode main.tex; \
		else \
			echo "WARNING: bibtex not found. References/Bibliography may not appear correctly."; \
		fi; \
	fi; \
	echo "Cleaning up intermediate files..."; \
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.lof main.log main.lot main.out main.toc main.bcf main.run.xml main.synctex.gz; \
	find . -type f -name "*.aux" -delete

# Check for common LaTeX mistakes before building
# Catches unescaped underscores inside \texttt{} — use \_ instead of _
check_latex:
	@echo "Checking for unescaped underscores inside \\texttt{}..."
	@python3 -c "\
import re, sys, glob; \
pat = re.compile(r'\\\\texttt\{([^}]*)\}'); \
errors = []; \
[errors.append(f'{f}:{i}: {l.rstrip()}') \
  for pattern in ['FEUP_Dissertation_format/body/*.tex', 'FEUP_Dissertation_format/frontmatter/*.tex'] \
  for f in glob.glob(pattern) \
  for i, l in enumerate(open(f), 1) \
  for m in pat.finditer(l) \
  if re.search(r'(?<!\\\\)_', m.group(1))]; \
(print('ERROR: Unescaped _ inside \\\\texttt{} — use \\\\_ instead:\n' + chr(10).join('  ' + e for e in errors)) or sys.exit(1)) if errors else None \
" 2>&1 || exit 1
	@echo "  OK — no unescaped underscores found."

# Build the dissertation report
dissertation: check_latex
	@echo "Building Dissertation..."
	@cd dissertation && \
	if command -v latexmk >/dev/null 2>&1; then \
		echo "Found latexmk, using it..."; \
		latexmk -pdf -interaction=nonstopmode main.tex; \
	else \
		echo "latexmk not found. Falling back to manual build sequence."; \
		pdflatex -interaction=nonstopmode main.tex; \
		if command -v bibtex >/dev/null 2>&1; then \
			echo "Found bibtex, running bibliography..."; \
			bibtex main; \
			pdflatex -interaction=nonstopmode main.tex; \
			pdflatex -interaction=nonstopmode main.tex; \
		else \
			echo "WARNING: bibtex not found. References/Bibliography may not appear correctly."; \
		fi; \
	fi; \
	echo "Cleaning up intermediate files..."; \
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.lof main.log main.lot main.out main.toc main.bcf main.run.xml main.synctex.gz; \
	find . -type f -name "*.aux" -delete

# Build the new FEUP Dissertation format template
# Logic: Check for latexmk -> if yes, use it.
#        If no, fall back to pdflatex + bibtex + pdflatex sequence.
feup_thesis: check_latex
	@echo "Building FEUP Dissertation (new template)..."
	@cd FEUP_Dissertation_format && \
	if command -v latexmk >/dev/null 2>&1; then \
		echo "Found latexmk, using it (force-complete mode)..."; \
		latexmk -pdf -f -interaction=nonstopmode main.tex; \
	else \
		echo "latexmk not found. Falling back to manual build sequence."; \
		pdflatex -interaction=nonstopmode main.tex; \
		if command -v biber >/dev/null 2>&1; then \
			echo "Found biber, running bibliography..."; \
			biber main; \
			pdflatex -interaction=nonstopmode main.tex; \
			pdflatex -interaction=nonstopmode main.tex; \
		else \
			echo "WARNING: biber not found. References/Bibliography may not appear correctly."; \
		fi; \
	fi; \
	echo "Cleaning up intermediate files..."; \
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.lof main.log main.lot main.out main.toc main.bcf main.run.xml main.synctex.gz missfont.log; \
	find . -type f -name "*.aux" -delete

# Clean up build files in all targets
clean:
	@echo "Cleaning up dissertation-preparation..."
	@cd dissertation-preparation && \
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C; \
	else \
		rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.bcf *.run.xml *.bbl *.blg *.synctex.gz main.pdf; \
	fi
	@echo "Cleaning up dissertation..."
	@cd dissertation && \
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C; \
	else \
		rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.bcf *.run.xml *.bbl *.blg *.synctex.gz main.pdf; \
	fi
	@echo "Cleaning up FEUP_Dissertation_format..."
	@cd FEUP_Dissertation_format && \
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C; \
	else \
		rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.bcf *.run.xml *.bbl *.blg *.synctex.gz main.pdf; \
	fi
	@find FEUP_Dissertation_format -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.blg" \) -delete
