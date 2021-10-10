from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'pyPDAG'
LONG_DESCRIPTION = """Python version for Partion-DAG as outlined in https://arxiv.org/pdf/1902.05173.pdf
"""

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pyPDAG", 
        version=VERSION,
        author="Syed Rahman",
        author_email="<syedhr264@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'matplotlib==3.4.2',
            'multiprocess==0.70.12.2',
            'munkres==1.1.4',
            'networkx==2.5',
            'numba==0.53.1',
            'numpy==1.20.3',
            'pandas==1.3.3',
            'scipy==1.7.1'
        ],
        # needs to be installed along with your package. Eg: 'caer'
        keywords=['python', 'pdag'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
