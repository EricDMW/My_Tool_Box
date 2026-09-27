# My Tool Box User Manual

This directory contains the LaTeX sources of the user manual for My Tool Box
1.0, the multi-agent reinforcement-learning environments (`env_lib`) and
research utilities (`toolkit`) of this repository.

## Building

Requirements: a TeX distribution that provides `pdflatex` and `latexmk`, for
example TeX Live or MacTeX. On Debian and Ubuntu:

```bash
sudo apt-get install latexmk texlive-latex-extra texlive-fonts-recommended lmodern
```

Build from this directory:

```bash
./build.sh
```

`build.sh` runs

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output main.tex
```

and writes the PDF to `output/main.pdf`. `latexmk` repeats the `pdflatex`
runs until all cross references are resolved. Remove the build products with

```bash
latexmk -C -outdir=output
```

The `output/` directory is ignored by Git. A pre-built copy of the manual is
committed as `main.pdf`; after editing the sources, rebuild and copy
`output/main.pdf` over it.

## Structure

```
docs/manual/
  main.tex                  preamble, macros and chapter list
  build.sh                  build script (latexmk)
  chapters/
    introduction.tex        overview, design principles, licence and citation
    installation.tex        requirements, extras, verification, building the manual
    envlib_overview.tex     common environment interface, registry, rendering
    kos_env.tex             Kuramoto oscillator networks (NumPy and PyTorch)
    networked_envs.tex      LineMsg and WirelessComm
    pistonball_env.tex      Pistonball
    consensus_env.tex       consensus and formation control
    ajlatt_env.tex          multi-robot localisation and target tracking
    toolkit_overview.tex    toolkit structure and parakit
    plotkit.tex             plotting
    neural_toolkit.tex      PyTorch networks and tabular tools
    examples.tex            example scripts, recording, benchmarks, contributing
    troubleshooting.tex     common problems and solutions
  appendices/
    api_reference.tex       signatures of the public API
    migration.tex           migrating code written for versions before 1.0
  figures/                  images included by the chapters
  output/                   build products (not tracked)
```

`main.tex` includes the chapters with `\include` in the order of the list
above, grouped into the parts Environments, Toolkit and Practice, followed by
the appendices.

## Writing conventions

- Chapters start with `\chapter{...}\label{chap:...}`; appendices use
  `app:...`, sections `sec:...`, tables `tab:...` and figures `fig:...`.
- Inline markup macros defined in `main.tex`: `\code{...}` for identifiers
  (escape underscores as `\_`), `\pkg{...}` for package and module names, and
  `\envid{...}` for registered environment ids.
- Code listings use the environments `pycode` (Python), `shellcode` (shell
  commands) and `textcode` (plain text); their content is written verbatim.
- Figures are placed in `figures/` and included by file name, since
  `\graphicspath` points to that directory. `\figwidth` is the default figure
  width.
- Use plain ASCII in the sources (LaTeX math for formulas) and no emojis or
  decorative symbols.
- Code examples must run against the current API. Check them in a headless
  environment (`MPLBACKEND=Agg`) before committing, and check the build log for
  errors, undefined references and overfull boxes.

## Licence

The manual is part of the My Tool Box project and is distributed under the
same MIT licence as the code (see `LICENSE` at the repository root).
