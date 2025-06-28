from setuptools import setup, find_packages

setup(
    name="toolkit",
    version="0.2.0",
    description="A comprehensive, research-quality plotting toolkit for reinforcement learning and research applications.",
    long_description=open("plotkit/README.md").read(),
    long_description_content_type="text/markdown",
    author="Dongming Wang",
    author_email="dongming.wang@email.edu",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "numpy>=1.20.0"
    ],
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="plotting, visualization, research, reinforcement-learning, matplotlib, seaborn",
    project_urls={
        "Bug Reports": "https://github.com/DMWang/toolkit/issues",
        "Source": "https://github.com/DMWang/toolkit",
        "Documentation": "https://github.com/DMWang/toolkit#readme",
    },
) 