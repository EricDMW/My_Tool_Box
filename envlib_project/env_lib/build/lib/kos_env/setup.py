from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kos_env",
    version="0.1.0",
    author="Dongming Wang",
    author_email="wdong025@ucr.edu",
    description="A Gymnasium environment for simulating and controlling Kuramoto oscillator synchronization dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kos-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "gymnasium.envs": [
            "kos_env = kos_env.__init__:register",
        ],
    },
    include_package_data=True,
    package_data={
        "kos_env": ["*.md", "*.txt"],
    },
    keywords=[
        "reinforcement-learning",
        "gymnasium",
        "kuramoto",
        "oscillator",
        "synchronization",
        "physics",
        "neuroscience",
        "complex-systems",
        "pytorch",
        "numpy",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/kos-env/issues",
        "Source": "https://github.com/yourusername/kos-env",
        "Documentation": "https://kos-env.readthedocs.io/",
    },
    zip_safe=False,
) 