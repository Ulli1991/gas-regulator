from setuptools import setup, find_packages

setup(
    name="gas-regulator",
    version="0.1.0",
    description="Gas-regulator model from Carr et al. 2023 for star formation regulation via hot CGM",
    author="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.3",
        "astropy>=4.0",
    ],
    python_requires=">=3.7",
)
