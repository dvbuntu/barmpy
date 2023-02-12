
BARN
====

One of the primary ensembles available in BARMPy (currently the only one) is that of the single hidden layer Neural Network.  When you collect several of these NNs, you have Bayesian Additive Regression Networks (BARN)

NN
--

Before building an ensemble, it's helpful to understand the core component that goes into it.  In this case, we use a neural network implemented in `sklearn` with a few extra methods to ease their use in BARN.  Eventually, this class will inherit from an abstract one for general BARM components.

.. automodule:: barmpy.barn.NN
   :members:

BARN Class
----------

Armed with a core `NN` class above, we can train the entire ensemble following our Bayesian procedure. 

.. automodule:: barmpy.barn.BARN
   :members:

Example
~~~~~~~

Let's walk through a minimal example training an ensemble with BARN.  Start by generating some data (or reading in some of your own).

.. code-block:: python

   import numpy as np
   X = np.random([100,2])
   # make a simply linear relationship
   Y = X[:,0] + 2*X[:,1] + np.random.random(100)/10

Now we'll initialize a `BARN` setup with 3 `NN`'s.  We'll use the default

.. code-block:: python

   from barmpy.barn import BARN, NN
   model = BARN(num_nets=3, dname='example')
   model.setup_nets()


Actually running the model is simple, but you can tweak the MCMC parameters to your liking.  After the specified number of MCMC iterations, your model is ready for pointwise inference by using the last ensemble in the chain.

.. code-block:: python

   model.train(X,Y, total_iters=100)
   Yhat = model.predict(X)
   print((Y-Yhat)**2/np.std(Y)) # relative error

Coming Soon
~~~~~~~~~~~

* Visualization example
* Tweaking MCMC parameters
* Hyperparameter tuning with cross-validation
