"""
Setup script for the Line Message Environment package.
"""

from setuptools import setup, find_packages

setup(
    name="linemsg_env",
    version="0.1.0",
    description="A standard gym environment for multi-agent line message passing",
    author="DSDP Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning multi-agent gymnasium environment",
    project_urls={
        "Source": "https://github.com/your-repo/linemsg_env",
        "Documentation": "https://github.com/your-repo/linemsg_env#readme",
    },
) 