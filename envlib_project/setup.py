from setuptools import setup, find_packages

setup(
    name="env_lib",
    version="0.1.0",
    description="A collection of multi-agent RL environments: kos_env, linemsg_env, pistonball_env, wireless_comm_env.",
    author="Dongming Wang",
    author_email="wdong025@ucr.edu",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "gymnasium",
        "gymnasium-robotics",
        # Add other dependencies as needed
    ],
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 