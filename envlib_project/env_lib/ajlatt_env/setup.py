#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for TTENV - Target Tracking Environment
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ttenv",
    version="0.1.0",
    author="Dongming Wang",
    author_email="dongming.wang@email.ucr.edu",
    description="A multi-robot target tracking environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ttenv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gym>=0.21.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "torch>=1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    include_package_data=True,
    package_data={
        "ttenv": ["maps/*.yaml", "maps/*.cfg", "maps/lib_obstacles/*.npy", "maps/lib_obstacles_2/*.npy"],
    },
) 