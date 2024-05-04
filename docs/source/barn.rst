
BARN
====

One of the primary ensembles available in BARMPy (currently the only one) is a single hidden layer Neural Network.  When you collect several of these NNs, you have Bayesian Additive Regression Networks (BARN).

NN
--

Before building an ensemble, it's helpful to understand the core component that goes into it.  In this case, we use a neural network implemented in `sklearn` with a few extra methods to ease their use in BARN.  Eventually, this class will inherit from an abstract one for general BARM components.

.. autoclass:: barmpy.barn.NN
   :members:
   :undoc-members:

BARN Class
----------

Equipped with a core `NN` class above, we can train the entire ensemble following our Bayesian procedure.

.. autoclass:: barmpy.barn.BARN_base
   :members:
   :undoc-members:

Both `barmpy.barn.BARN` (for regression) and `barmpy.barn.BARN_bin` (for binary classification) inherit from `barmpy.barn.BARN_base`.

Example
~~~~~~~

Let's walk through a minimal example training an ensemble with BARN.  Start by generating some data (or reading in some of your own).

.. code-block:: python

   import numpy as np
   X = np.random.random([100,2])
   # make an ordinary linear relationship
   Y = X[:,0] + 2*X[:,1] + np.random.random(100)/10

Now we'll initialize a `BARN` setup with 3 `NN`'s.  We'll use the default

.. code-block:: python

   from barmpy.barn import BARN, NN
   model = BARN(num_nets=3, dname='example', epochs=100)


Actually running the model is straightforward, but you can tweak the MCMC parameters to your liking.  After the specified number of MCMC iterations, your model is ready for pointwise inference by using the last ensemble in the chain.

.. code-block:: python

   model.fit(X,Y)
   Yhat = model.predict(X)
   print((Y-Yhat)**2/np.std(Y)) # relative error

Custom Callback Example
~~~~~~~~~~~~~~~~~~~~~~~

BARMPy also support custom model callbacks.  Callbacks are a way to run a routine in between MCMC iterations.  This is typically done to either log information or check for an early stopping condition.  We provide several callbacks in the library itself, though you can supply your own function as well.  We recommend `barmpy.barn.BARN_base.improvement` to check for early stopping with validation data (note such data will also be used for MCMC acceptance, but not NN training, if provided).  The set of all provided callbacks are:

* `barmpy.barn.BARN_base.improvment` - Check if validation error has stopped improving, indicating model has started to overfit and training should stop
* `barmpy.barn.BARN_base.rfwsr` - Relative fixed-width stopping rule to check if MCMC estimate has converged
* `barmpy.barn.BARN_base.stable_dist` - Check if distribution of neuron counts is stable, indicating stationary distribution reached
* `barmpy.barn.BARN_base.trans_enough` - Check if enough transitions were accepted to justify additional MCMC iterations toward stationary posterior

To use a callback, we need to add it to a dictionary and pass that to the `callbacks` argument of `barmpy.barn.BARN` (or `barmpy.barn.BARN_bin`, as appropriate).  The key to the dictionary should be the Python function or method itself, while the values are additional arguments provided to that function.  Each iteration, we will call each callback function, passing the ensemble itself as the first argument (to enable access to its internals).

Here is a small snippet showing how to use the `stable_dist` callback:

.. code-block:: python

   callbacks = {barmpy.barn.BARN.stable_dist:
                   {'check_every':1,
                   'skip_first':4}}
   model = BARN(num_nets=10,
               callbacks=callbacks,
               n_iter=n_iter)

CV Tuning Example
~~~~~~~~~~~~~~~~~

BARN is implemented as an `sklearn`_ class, meaning we can use standard `sklearn`_ methods like `GridSearchCV` to tune the hyperparameters for the best possible result.  This does take considerably more processing power to test the various parameter configurations, so be mindful when considering the number of possible hyperparameter values.

Much like `BART`_, we apply cross-validated hyperparameter tuning to set the priors (i.e. the expected number of neurons in a network, `l`).  But as with BART, we do not seek an *exact* match, only something that generally agrees with the data.  Below is a short series of examples using various `sklearn`_ approaches.

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

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

Though BARN is implemented as an `sklearn` regression class and you can use it with any compatible visualization library, we also have some built-in methods.  The first is `model.viz`.  After training, this creates a plot of predicted vs target values, both for initial BARN (i.e. before training) and final results.

.. code-block:: python

   from sklearn import datasets
   from sklearn.model_selection import train_test_split
   from barmpy.barn import BARN, NN
   import numpy as np
   from sklearn.decomposition import PCA
   from sklearn.preprocessing import StandardScaler

   db = datasets.load_diabetes()
   Xtr, Xte, Ytr, Yte = train_test_split(db.data, db.target, test_size=0.2, random_state=0)

   # rescale inputs with PCA (and output normalized)
   Xtr_o = np.copy(Xtr)
   Xte_o = np.copy(Xte)
   scale_x = PCA(n_components=Xtr.shape[1], whiten=False)
   scale_x.fit(Xtr)
   Xtr = scale_x.transform(Xtr_o)
   Xte = scale_x.transform(Xte_o)
   Ytr_o = np.copy(Ytr)
   Yte_o = np.copy(Yte)
   scale_y = StandardScaler() # no need to PCA
   scale_y.fit(Ytr.reshape((-1,1)))
   Ytr = scale_y.transform(Ytr_o.reshape((-1,1))).reshape(-1)
   Yte = scale_y.transform(Yte_o.reshape((-1,1))).reshape(-1)

   model = BARN(num_nets=10, dname='example',
      l=1,
      act='logistic',
      epochs=100,
      n_iter=100)
   model.fit(Xtr, Ytr, Xte=Xte, Yte=Yte)
   model.viz(outname='viz_test.png', initial=True)

.. image:: viz_test.png
  :width: 600
  :alt: Left scatterplot shows initial BARN results.  Prediction vs Target values look like flat horizontal line.  Training R2 is 0.07 while test RMSE is 0.88.  On the right is a similar scatterplot showing trained BARN results.  The points start to track the goal 1-1 correspondence.  Training R2 is 0.56 while test RMSE is 0.76, an improvement.

We also have tool to view the validation error progression over the MCMC iterations.  This can be helpful to assess convergence.  Reusing the above model, it looks like this particular BARN model isn't converging well.

.. code-block:: python

   model.phi_viz(outname='', close=False)

.. image:: phi_viz.png
  :width: 600
  :alt: Line plot showing error of each MCMC iteration ensemble.  Error bounces up and down, not clearly converging.

Coming Soon
~~~~~~~~~~~

* Tweaking MCMC parameters

.. _sklearn: https://scikit-learn.org/
.. _sklearn.model_selection: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
.. _BART: https://arxiv.org/abs/0806.3286
