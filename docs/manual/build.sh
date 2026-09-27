#!/usr/bin/env bash
# Build the manual into output/main.pdf (requires a TeX distribution with latexmk).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output main.tex
echo "Manual written to $(pwd)/output/main.pdf"
