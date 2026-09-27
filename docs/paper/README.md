# Paper

`paper.tex` introduces the `env_lib` environments, how to use them and the
integrated baselines (classical controllers and the `marl_algorithms`
learning baselines) in the two-column format of the ICML author template.

- `my_tool_box_paper.pdf`: named preprint.
- `my_tool_box_paper_anonymous.pdf`: anonymous version for double-blind
  review (authors, affiliation and repository link replaced by placeholders).

## Building

Requirements: a TeX distribution with `latexmk`, `pgfplots` and the Times
fonts (TeX Live: `texlive-latex-extra`, `texlive-fonts-recommended`,
`texlive-pictures`, `texlive-science`).

```bash
./build.sh               # output/paper.pdf, named preprint
./build.sh --anonymous   # output/paper_anonymous.pdf, anonymous submission
```

## ICML style files

The official ICML style files are distributed by the conference and are not
included. `icmllike.sty` reproduces the layout of the ICML template (letter
paper, 6.75 x 9 in two-column body, Times, title between rules, author block
and running title) and the same author macros, so the paper builds without
them. For a submission, download the style files of the target year and
replace in `paper.tex`

```latex
\usepackage[\icmloptions]{icmllike}   ->   \usepackage[\icmloptions]{icml20XX}
\bibliographystyle{plainnat}          ->   \bibliographystyle{icml20XX}
```

The official style adds line numbers and the page-limit conventions of the
year; check the page count again after the switch (the main text currently
ends on page 7, followed by references and the appendix).

## Sources of the numbers

Every table and figure is produced by a script of the repository; Appendix C
of the paper lists the commands. Figures are copied from
`docs/manual/figures/` and `docs/images/`.
