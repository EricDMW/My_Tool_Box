# My Tool Box User Manual

This directory contains the comprehensive user manual for the My Tool Box project, written in LaTeX format.

## Structure

```
manual/
├── main.tex                 # Main LaTeX document
├── build.sh                 # Build script
├── README.md               # This file
├── chapters/               # Individual chapters
│   ├── introduction.tex    # Introduction and overview
│   ├── installation.tex    # Installation and setup
│   ├── toolkit_overview.tex # Toolkit overview
│   ├── neural_toolkit.tex  # Neural toolkit documentation
│   ├── plotkit.tex         # Plotkit documentation
│   ├── envlib_overview.tex # Environment library overview
│   ├── pistonball_env.tex  # Pistonball environment
│   ├── kos_env.tex         # Kuramoto oscillator environment
│   ├── examples.tex        # Examples and tutorials
│   └── troubleshooting.tex # Troubleshooting guide
├── appendices/             # Appendices
│   └── api_reference.tex   # API reference
├── figures/                # Figures and images
└── output/                 # Generated PDF output
```

## Building the Manual

### Prerequisites

1. **LaTeX Distribution**: Install a LaTeX distribution
   - **Ubuntu/Debian**: `sudo apt-get install texlive-full`
   - **macOS**: `brew install --cask mactex`
   - **Windows**: Install MiKTeX or TeX Live

2. **Build Tools**: Ensure `pdflatex` and `makeindex` are available

### Building

1. **Navigate to the manual directory**:
   ```bash
   cd manual
   ```

2. **Run the build script**:
   ```bash
   ./build.sh
   ```

3. **Check the output**: The generated PDF will be in `output/main.pdf`

### Manual Build

If you prefer to build manually:

```bash
# First compilation
pdflatex -output-directory=output main.tex

# Generate index (if makeindex is available)
makeindex -o output/main.ind output/main.idx

# Second compilation
pdflatex -output-directory=output main.tex

# Third compilation (for references)
pdflatex -output-directory=output main.tex
```

## Manual Contents

### Chapters

1. **Introduction** - Overview of the My Tool Box project
2. **Installation** - Setup and installation instructions
3. **Toolkit Overview** - Overview of the unified toolkit package
4. **Neural Toolkit** - Comprehensive neural network documentation
5. **Plotkit** - Plotting utilities documentation
6. **Environment Library Overview** - Overview of reinforcement learning environments
7. **Pistonball Environment** - Multi-agent physics environment
8. **Kuramoto Oscillator Environment** - Complex systems environment
9. **Examples** - Comprehensive examples and tutorials
10. **Troubleshooting** - Common issues and solutions

### Appendices

- **API Reference** - Complete API documentation for all components

## Features

- **Comprehensive Coverage**: Complete documentation of all toolkit components
- **Code Examples**: Extensive code examples for all features
- **Troubleshooting**: Solutions to common problems
- **API Reference**: Complete API documentation
- **Professional Format**: High-quality LaTeX formatting
- **Index**: Automatic index generation
- **Cross-references**: Internal links and references

## Customization

### Adding New Chapters

1. Create a new `.tex` file in the `chapters/` directory
2. Add the chapter to `main.tex`:
   ```latex
   \input{chapters/your_chapter.tex}
   ```

### Modifying Styles

Edit the preamble in `main.tex` to customize:
- Page layout
- Fonts and colors
- Code listing styles
- Figure and table styles

### Adding Figures

1. Place images in the `figures/` directory
2. Reference them in your chapters:
   ```latex
   \begin{figure}[H]
   \centering
   \includegraphics[width=0.8\textwidth]{figures/your_image.png}
   \caption{Your caption}
   \label{fig:your_label}
   \end{figure}
   ```

## Troubleshooting

### Build Issues

1. **Missing LaTeX packages**: Install required packages
   ```bash
   sudo apt-get install texlive-latex-extra
   ```

2. **Font issues**: Install additional fonts
   ```bash
   sudo apt-get install texlive-fonts-extra
   ```

3. **Memory issues**: Increase LaTeX memory limits
   ```bash
   export TEXMFVAR=/tmp/texmf
   ```

### Common Problems

- **Long build times**: Use `-interaction=nonstopmode` flag
- **Missing references**: Run multiple compilation passes
- **Index issues**: Ensure `makeindex` is installed

## Contributing

When contributing to the manual:

1. **Follow LaTeX conventions**: Use proper LaTeX syntax and structure
2. **Include examples**: Provide working code examples
3. **Update index**: Add new terms to the index when appropriate
4. **Test builds**: Ensure the manual builds correctly
5. **Check references**: Verify all cross-references work

## License

This manual is part of the My Tool Box project and follows the same license terms.

## Support

For issues with the manual:

1. Check the troubleshooting section
2. Review the build script output
3. Ensure all dependencies are installed
4. Check LaTeX distribution compatibility

## Version History

- **v1.0** - Initial manual release
- Comprehensive coverage of toolkit and environment library
- Professional LaTeX formatting
- Complete API reference
- Extensive examples and troubleshooting 