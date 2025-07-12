#!/bin/bash

# Clean script for LaTeX auxiliary files
# This script removes temporary files generated during PDF compilation

echo "🧹 Cleaning LaTeX auxiliary files..."

# Remove LaTeX auxiliary files
rm -f *.aux
rm -f *.log
rm -f *.out
rm -f *.toc
rm -f *.lof
rm -f *.lot
rm -f *.fls
rm -f *.fdb_latexmk
rm -f *.synctex.gz
rm -f *.bbl
rm -f *.blg
rm -f *.idx
rm -f *.ind
rm -f *.ilg
rm -f *.glo
rm -f *.gls
rm -f *.glg
rm -f *.acn
rm -f *.acr
rm -f *.alg
rm -f *.ist
rm -f *.nav
rm -f *.snm
rm -f *.vrb

# Remove auxiliary files from subdirectories
find . -name "*.aux" -type f -delete
find . -name "*.log" -type f -delete
find . -name "*.out" -type f -delete
find . -name "*.toc" -type f -delete
find . -name "*.lof" -type f -delete
find . -name "*.lot" -type f -delete
find . -name "*.fls" -type f -delete
find . -name "*.fdb_latexmk" -type f -delete
find . -name "*.synctex.gz" -type f -delete

# Remove output directory if it exists
if [ -d "output" ]; then
    echo "🗑️  Removing output directory..."
    rm -rf output
fi

# Remove any temporary files
rm -f *~
rm -f .DS_Store
rm -f Thumbs.db

echo "✅ Cleanup completed!"
echo "📁 Removed auxiliary files and temporary files" 