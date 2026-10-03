#!/usr/bin/env bash
# Build the paper into output/ (requires a TeX distribution with latexmk,
# pgfplots and the Times fonts). Pass --anonymous for the anonymous
# submission version (authors and repository link hidden).
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--anonymous" ]]; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output \
    -jobname=paper_anonymous -usepretex='\def\icmloptions{}' paper.tex
  echo "Paper written to $(pwd)/output/paper_anonymous.pdf"
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output paper.tex
  echo "Paper written to $(pwd)/output/paper.pdf"
fi
