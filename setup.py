from setuptools import setup, find_packages

setup(
    name="sdmose",
    version="0.1.0",
    description="Scientific Discovery - Mixture of Scientific Experts for Climate Equation Discovery",
    author="Shlok Vaishnav",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "scikit-learn",
        "pysr",
        "torch",
        "hydra-core",
    ],
)
