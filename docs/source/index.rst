.. BARMPy documentation main file, created by
   sphinx-quickstart on Sat Sep 15 12:15:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BARMPy's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   math
   barn


BARMPy is a Python library to implement `Bayesian Additive Regression Networks <https://arxiv.org/abs/2404.04425>`_ (BARN, sometimes generalized to BAR*M* for generic *Models*), a machine learning method to ensemble together small models using Markov Chain Monte Carlo (MCMC) methods.  This approach takes much inspiration from `Bayesian Additive Regression Trees <https://arxiv.org/abs/0806.3286>`_ (BART).

BARN excels at accurately modeling a wide variety of regression problems with equal aplomb.  This does come with a cost, as running the MCMC iterations and training neural networks can be computationally intensive.  Problems with a thousand data points and ten features can take on the order of a second, but this varies with *difficulty* of the problem as much as the problem size itself.

Installation
============

PyPi packages will be coming soon, but until then, you can install from the repository itself:

.. code-block:: bash

   # Current release
   pip install barmpy
   # Latest development version
   git clone git@github.com:dvbuntu/barmpy.git
   pip install -e ./barmpy

Tutorial
========

Currently available in `BARMPy Repo <https://github.com/dvbuntu/barmpy/blob/main/examples/tutorial.Rmd>`_ or precompiled `here <https://drive.google.com/file/d/1FgpCyEUqqnihkfm-6nuV5RdZwAJlSJq5/view?usp=drive_link>`_.

Links
=====

* :ref:`genindex`
* :ref:`search`
* `GitHub Repo <https://github.com/dvbuntu/barmpy>`_
* `PyPi <https://pypi.org/project/barmpy/>`_

