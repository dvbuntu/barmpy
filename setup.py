from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='barmpy',
    version='1.1.0',
    description='Bayesian Additive Regression Models implementation for Python',
    url='https://github.com/dvbuntu/barmpy',
    author='Danielle Van Boxel',
    author_email='vanboxel@arizona.edu',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'tqdm',
    ]
)
