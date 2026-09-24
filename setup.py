from setuptools import setup, find_packages

setup(
    name="opensetdga",
    version="1.0.0",
    description="OpenSetDGA: A Benchmark for Open-Set DGA Detection",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "lightgbm>=4.0",
        "torch>=2.0",
        "tldextract>=3.4",
    ],
)
