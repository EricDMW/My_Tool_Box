from setuptools import setup, find_packages
import os

setup(
    name="toolkit",
    version="0.2.0",
    description="A comprehensive research toolkit for reinforcement learning and research applications, including neural networks and plotting utilities.",
    long_description=open("toolkit/README.md").read() if os.path.exists("toolkit/README.md") else "A comprehensive research toolkit for reinforcement learning and research applications.",
    long_description_content_type="text/markdown",
    author="Dongming Wang",
    author_email="dongming.wang@email.edu",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0"
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
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="neural-networks, plotting, visualization, research, reinforcement-learning, deep-learning, machine-learning, matplotlib, seaborn, pytorch, tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/DMWang/toolkit/issues",
        "Source": "https://github.com/DMWang/toolkit",
        "Documentation": "https://github.com/DMWang/toolkit#readme",
    },
) 