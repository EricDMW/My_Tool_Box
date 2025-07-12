#!/bin/bash

# Build script for the LaTeX manual
# This script compiles the main.tex file into a PDF

# Create output directory if it doesn't exist
mkdir -p output

# First compilation
echo "First compilation..."
pdflatex -output-directory=output -interaction=nonstopmode main.tex

# Generate index if makeindex is available
if command -v makeindex &> /dev/null; then
    echo "Generating index..."
    makeindex -o output/main.ind output/main.idx
else
    echo "makeindex not found, skipping index generation"
fi

# Second compilation
echo "Second compilation..."
pdflatex -output-directory=output -interaction=nonstopmode main.tex

# Third compilation (for references)
echo "Third compilation..."
pdflatex -output-directory=output -interaction=nonstopmode main.tex

# Check if PDF was created
if [ -f "output/main.pdf" ]; then
    echo "✅ Manual successfully compiled to output/main.pdf"
    echo "📄 PDF size: $(du -h output/main.pdf | cut -f1)"
else
    echo "❌ Compilation failed. Check the log files in output/ directory"
    exit 1
fi

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f output/*.aux output/*.log output/*.out output/*.toc output/*.lof output/*.lot output/*.idx output/*.ind output/*.ilg output/*.glo output/*.gls output/*.glg output/*.bbl output/*.blg output/*.fdb_latexmk output/*.fls output/*.synctex.gz

echo "🎉 Build completed successfully!"
echo "📖 Manual is available at: output/main.pdf" 