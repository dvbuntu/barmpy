.. role:: m(math)
.. default-role:: math

Math
====

We briefly describe some of the mathematics behind Bayesian Additive Regression Models like BART and BARN.  For more technical details, please see `Bayesian Additive Regression Trees <https://arxiv.org/abs/0806.3286>`_ and `Bayesian Additive Regression Networks <https://arxiv.org/abs/2404.04425>`_.  

First, we focus on "regression" problems, where there is floating point output, `y`, and a vector of floating point inputs, `x`, for each data point.  In general, we posit that there exists some function, `u`, such that `y = u(x) + \epsilon`, where `\epsilon` is a noise term.  Our objective is to find some *other* function, `f`, so that `f(x) \approx u(x)`, because if we knew `u`, we'd be done already.

An approach like linear regression assumes `f(x) = \sum w_j x_j`, i.e. a weighted sum of the inputs.  In ordinary linear regression (OLS), we find the weights, `w_j`, that minimize a loss function, `L = \sum_i (y_i - f(x_i))^2`, also called the "squared error loss".  If we average `L` over the number of data points, when we have a "mean squared error" or MSE.

Bayesian Additive Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our methods like BARN also seek to minimize the MSE, but we posit a different form for `f`, so we need a different optimization procedure.  We will in fact learn `k` different functions, `f_1,f_2,\ldots, f_k` and aggregate them so `f(x) = \sum_j f_j(x)`.  This is known as model ensembling.

Well, so what does each `f_j` look like, and how do we find them all?  In BART, each `f_j` is a "Decision Tree"; it asks a series of questions about the input data in order to arrive at an output result.  In BARN, however, `f_j` is a small single hidden layer neural network.  While there exist standard methods to train both of these, we're going to do something somewhat more involved.

Really, there's many sets of such models (trees or NNs) that would probably accurately model our data.  But maybe we also have some initial ideas on how big these models should be, because we want to avoid overfitting.  So really, there's a probability distribution of models that both accurately model the data and account for our prior ideas.  This is exactly a Bayesian "posterior" estimate: the probability of the evidence under the model times the prior probability of the model.

.. math:: p(f|x,y) \propto p(y|f,x) p(f)

The trick is how to sample from this kind of space?  As it's some probability distribution, we can engineer a Markov Chain that has it as a stationary distribution so that sampling from the chain means we sample from the desired models.  This is known as "Markov Chain Monte Carlo".  There's still one issue, however, as in our ensemble, the dimensionality of the space is quite large.  So rather than sample all `f_1, f_2, \ldots, f_k` at once, we instead sample one at a time while holding the rest fixed.

Suppose we fix `f_2, \ldots, f_k` in whatever their current state is.  Now compute the "residual", `r_i = y_i - \sum_{j=2}^k f_j(x_i)`, which is just whatever is leftover of the target value after subtracting out the predictions of the frozen members of the ensemble.  This gives a new sort of a mini-regression problem with `x_i,r_i` and `f_1`.

*Now* we can setup an MCMC to find a better `f_1` sampling from the posterior.  This requires 3 essential components for both `M` (old) and `M'` (new):

#. `T(M'|M)`, transition proposal probability (what's the new model going to look like?)
#. `p(y|M',x)`, error probability (how well does this model fit the data) 
#. `p(M')`, prior probability (how likely is this type of model)

If we have these for both the old and new model, then we can compute an acceptance probability of replacing the old model with the new:

.. math:: \alpha = min(1, \frac{T(M'|M)p(y|M',x)p(M')}{T(M|M')p(y|M,x)p(M)})

If a uniform random number between 0 and 1 is less than this `\alpha`, then we set `f_1=M'`, else we leave it as `M'`.  In this case, we take just a single step; then we will fix `f_1`, free `f_2`, recompute the residual, and repeat the process.  Cycling through each of the models in the ensemble is one step in the MCMC process; we will probably go for hundreds of iterations.
