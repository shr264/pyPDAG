from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'pyPDAG'
LONG_DESCRIPTION = """Python version for Partion-DAG as outlined in https://arxiv.org/pdf/1902.05173.pdf
"""

setup(
    name="pyPDAG",
    version=VERSION,
    author="Syed Rahman",
    author_email="<syedhr264@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/shr264/pyPDAG/",  # Change this to your actual repo if applicable
    packages=find_packages(),  # Automatically finds 'pdag' and other subpackages
    package_dir={"pdag": "pdag"},  # Explicitly map the package directory
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "scipy",
        "statsmodels",
        "matplotlib",
        "numba",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version required
)
