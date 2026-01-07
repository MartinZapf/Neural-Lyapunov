"""Setup script for neural-lyapunov package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-lyapunov",
    version="1.0.0",
    author="Martin Zapf",
    description="Neural Lyapunov functions for sliding mode controllers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MartinZapf/Neural-Lyapunov",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "hpo": [
            "optuna>=3.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "optuna>=3.0.0",
        ],
    },
)
