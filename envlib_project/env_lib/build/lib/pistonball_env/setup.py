"""
Setup script for the Pistonball environment.

This script helps install the environment and its dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="pistonball_env",
    version="1.0.0",
    description="A standard gym environment for the multi-agent pistonball game",
    author="Dongming Wang",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.0",
        "pygame>=2.0.0",
        "pymunk>=6.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 