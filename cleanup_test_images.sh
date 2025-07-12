#!/bin/bash

# Cleanup script for test_*.png files
# This script deletes all files that start with "test_" and end with ".png"

echo "=========================================="
echo "Test Image Cleanup Script"
echo "=========================================="

# Check if there are any matching files
if ! ls test_*.png 1> /dev/null 2>&1; then
    echo "No test_*.png files found in the current directory."
    exit 0
fi


# Count the number of matching files
file_count=$(ls test_*.png 2>/dev/null | wc -l)
echo "Found $file_count file(s) matching pattern 'test_*.png':"
echo ""

# List all matching files
ls -la test_*.png
echo ""

# Ask for user confirmation
read -p "Do you want to delete these files? (y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo "Deleting files..."
    rm test_*.png
    echo "✅ Successfully deleted $file_count file(s)."
else
    echo "❌ Operation cancelled. No files were deleted."
fi

echo "==========================================" 