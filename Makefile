# Makefile for FEUP Thesis

# Defaults
.PHONY: all report preparation dissertation thesis clean check_deps

# Default target
all: report

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

# Build the dissertation report
dissertation:
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

# Placeholder for the future thesis report
thesis:
	@echo "Thesis report target is not implemented yet."

# Clean up build files
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
