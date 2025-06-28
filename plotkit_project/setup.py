from setuptools import setup, find_packages

setup(
    name="plotkit",
    version="0.1.0",
    description="A simple, extensible plotting toolkit for RL and research.",
    author="Dongming Wang",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "seaborn",
        "numpy"
    ],
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 