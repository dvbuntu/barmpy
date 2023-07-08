
BARN
====

One of the primary ensembles available in BARMPy (currently the only one) is that of the single hidden layer Neural Network.  When you collect several of these NNs, you have Bayesian Additive Regression Networks (BARN)

NN
--

Before building an ensemble, it's helpful to understand the core component that goes into it.  In this case, we use a neural network implemented in `sklearn` with a few extra methods to ease their use in BARN.  Eventually, this class will inherit from an abstract one for general BARM components.

.. autoclass:: barmpy.barn.NN
   :members:
   :undoc-members:

BARN Class
----------

Armed with a core `NN` class above, we can train the entire ensemble following our Bayesian procedure. 

.. autoclass:: barmpy.barn.BARN
   :members:
   :undoc-members:

Example
~~~~~~~

Let's walk through a minimal example training an ensemble with BARN.  Start by generating some data (or reading in some of your own).

.. code-block:: python

   import numpy as np
   X = np.random([100,2])
   # make an ordinary linear relationship
   Y = X[:,0] + 2*X[:,1] + np.random.random(100)/10

Now we'll initialize a `BARN` setup with 3 `NN`'s.  We'll use the default

.. code-block:: python

   from barmpy.barn import BARN, NN
   model = BARN(num_nets=3, dname='example')
   model.setup_nets()


Actually running the model is straightforward, but you can tweak the MCMC parameters to your liking.  After the specified number of MCMC iterations, your model is ready for pointwise inference by using the last ensemble in the chain.

.. code-block:: python

   model.train(X,Y, total_iters=100)
   Yhat = model.predict(X)
   print((Y-Yhat)**2/np.std(Y)) # relative error

CV Tuning Example
~~~~~~~~~~~~~~~~~

BARN is implemented as an `sklearn`_ class, meaning we can use standard `sklearn`_ methods like `GridSearchCV` to tune the hyperparameters for the best possible result.  This does take considerably more processing power to test the various parameter configurations, so be mindful when considering the number of possible hyperparameter values.

Much like `BART`_, we apply cross-validated hyperparameter tuning to set the priors (i.e. the expected number of neurons in a network, `l`).  But as with BART, we do not seek an *exact* match, only something that generally agrees with the data.  Below is a short series of examples using various `sklearn` approaches.

.. code-block:: python

   from sklearn import datasets
   from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
   from barmpy.barn import BARN
   db = datasets.load_diabetes()
   scoring = 'neg_root_mean_squared_error'

   # exhaustive grid search
   ## first make prototype with fixed parameters
   bmodel = BARN(num_nets=10,
             random_state=0,
             warm_start=True,
             solver='lbfgs')
   ## declare parameters to exhaust over
   parameters = {'l': (1,2,3)}
   barncv = GridSearchCV(bmodel, parameters,
                   refit=True, verbose=4,
                   scoring=scoring)
   barncv.fit(db.data, db.target)
   print(barncv.best_params_)

   # randomized search with distributions
   from sklearn.model_selection import RandomizedSearchCV
   from scipy.stats import poisson
   ## first make prototype with fixed parameters
   bmodel = BARN(num_nets=10,
             random_state=0,
             warm_start=True,
             solver='lbfgs')
   ## declare parameters and distributions
   parameters = {'l': poisson(mu=2)}
   barncv = RandomizedSearchCV(bmodel, parameters,
                   refit=True, verbose=4,
                   scoring=scoring, n_iter=3)
   barncv.fit(db.data, db.target)
   print(barncv.best_params_)

In particular, note the need to set the `scoring = 'neg_root_mean_squared_error'`, which is what we recommend for default regression problems.  You can find more scoring options at the `sklearn.model_selection`_ page.

Also, when using a method like `RandomizedSearchCV`, be careful to supply appropriate distributions.  Here, `l` takes discrete values, so we specify a discrete Poisson probability distribution to sample from.  Note, however, that this distribution is *not* the distribution BARN uses for internal MCMC transitions.  This distribution is only for CV sampling the prior parameters.

Coming Soon
~~~~~~~~~~~

* Visualization example
* Tweaking MCMC parameters

.. _sklearn: https://scikit-learn.org/
.. _sklearn.model_selection: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
.. _BART: https://arxiv.org/abs/0806.3286
