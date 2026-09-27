#!/usr/bin/env bash
# Build the introduction slides into output/slides.pdf (requires a TeX
# distribution with latexmk, beamer, the metropolis theme and pgfplots).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output slides.tex
echo "Slides written to $(pwd)/output/slides.pdf"
