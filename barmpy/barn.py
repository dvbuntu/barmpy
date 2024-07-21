try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# use sklearn for now, could upgrade to keras later if we want
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import sklearn.neural_network as sknn
from sklearn.utils import check_random_state as crs
import sklearn.model_selection as skms
import sklearn.metrics as metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pymc
import pickle
import warnings
HAVE_TF = True
try:
    import tensorflow as tf
    import tensorflow.keras.layers as tkl
    import tensorflow.keras.regularizers as tkr
except:
    HAVE_TF = False

INFO = np.iinfo(np.int32)
SMALL = INFO.min + 1
BIG = INFO.max - 1
REG = 0.01 # regularization
ACT = 'logistic'
#ACT = 'relu'

class NN(object):
    '''
    Neural Network with single hidden layer implemented with sklearn.

    Includes methods to do MCMC transitions and calculations.
    '''
    def __init__(self,
            num_nodes=10,
            weight_donor=None,
            l=2,
            lr=0.01,
            r=42,
            epochs=20,
            x_in=None,
            batch_size=512,
            solver=None,
            tol=1e-3,
            reg=REG,
            act=ACT,
            binary=False):
        self.num_nodes = num_nodes
        self.x_in = x_in
        # make an NN with a single hidden layer with num_nodes nodes
        ## can set max_iter to set max_epochs
        if solver is None:
            if num_nodes < 10:
                solver = 'lbfgs'
            else:
                solver = 'adam'
        # this binary is only for generic NN usage, not for BARN (which still uses regression networks underneath)
        if binary:
            mlp = sknn.MLPClassifier
        else:
            mlp = sknn.MLPRegressor
        self.model = mlp([num_nodes],
                learning_rate_init=lr,
                random_state=r,
                max_iter=epochs,
                batch_size=batch_size,
                warm_start=True,
                solver=solver,
                alpha=reg,
                activation=act,
                tol=tol) # TODO: scale with output
        # l is poisson shape param, expected number of nodes
        self.l = l
        self.lr = lr
        if r is None:
            r = int(time.time())
        self.r = r
        self.epochs = epochs
        self.reg = reg
        self.act = act
        if weight_donor is not None:
            # inherit the first num_nodes weights from this donor
            donor_num_nodes = weight_donor.num_nodes
            donor_weights, donor_intercepts = weight_donor.get_weights()
            self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)

    def save(self, fname):
        '''
        Save NN to disk as a NumPy archive
        '''
        params = np.array([self.num_nodes, self.l, self.lr, self.r,
            self.epochs, self.x_in, self.reg, self.act, self.binary])
        coefs_, intercepts_ = self.model.get_weights()
        np.savez_compressed(fname, params=params,
                coefs_=coefs_,
                intercepts_=intercepts_)

    def get_weights(self):
        '''
        Return `sklearn` style tuple of `(coefs_, intercepts_)`
        '''
        return (self.model.coefs_, self.model.intercepts_)

    def accept_donation(self, donor_num_nodes, donor_weights, donor_intercepts):
        '''
        Replace our weights with those of another `NN` (passed as weights).

        Donor can be different size; if smaller, earlier weights in donee
        are overwritten.
        '''
        # a bit of a workaround to create weight arrays and things
        num_nodes = self.num_nodes
        self.model._random_state = crs(self.r)
        self.model._initialize(np.zeros((1,1),dtype=donor_weights[0].dtype),
                               [donor_weights[0].shape[0], num_nodes, 1],
                               donor_weights[0].dtype)
        # TODO: use self.model.model_.set_weights()
        if donor_num_nodes == num_nodes:
            self.model.coefs_ = [d.copy() for d in donor_weights]
            self.model.intercepts_ = [d.copy() for d in donor_intercepts]
        elif donor_num_nodes > num_nodes:
            self.model.coefs_ = [donor_weights[0][:,:num_nodes].copy(),
                                 donor_weights[1][:num_nodes].copy()]
            self.model.intercepts_ = [donor_intercepts[0][:num_nodes].copy(),
                                 donor_intercepts[1].copy()]
        else:
            self.model.coefs_[0][:,:donor_num_nodes] = donor_weights[0].copy()
            self.model.coefs_[1][:donor_num_nodes] = donor_weights[1].copy()
            self.model.intercepts_[0][:donor_num_nodes] = donor_intercepts[0].copy()
            self.model.intercepts_[1] = donor_intercepts[1].copy()

    @staticmethod
    def load(fname):
        '''
        Read a NumPy archive of an NN (such as created by `save`) into a new NN
        '''
        network = np.load(fname)
        # enable loading old models, which were never binary
        if len(network['params']) == 8:
            network['params'] = network['params'].tolist() + [False]
        N = NN(network['params'][0],
               l=network['params'][1],
               lr=network['params'][2],
               r=network['params'][3],
               epochs=network['params'][4],
               x_in=network['params'][5],
               reg=network['params'][6],
               act=network['params'][7],
               binary=network['params'][8],
               )
        donor_num_nodes = N.num_nodes
        donor_weights = network['coefs_']
        donor_intercepts_ = network['intercepts_']
        self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)
        return N

    def train(self, X, Y):
        '''Train network from current position with given data'''
        self.model.fit(X,Y)

    def log_prior(self, pmf=scipy.stats.poisson.logpmf):
        '''
        Log prior probability of the `NN`.  Assumes one distribution
        parameter, `self.l`.
        '''
        return pmf(self.num_nodes, self.l)

    def log_likelihood(self, X, Y, std):
        '''
        Log likelihood of `NN` assuming normally distributed errors.
        '''
        # compute residuals
        yhat = np.squeeze(self.model.predict(X))
        resid = Y - yhat
        # compute stddev of these
        #std = np.std(resid) # maybe use std prior here?
        #std = self.std
        # normal likelihood
        return np.sum(scipy.stats.norm.logpdf(resid, 0, std))

    def log_acceptance(self, X, Y):
        '''Natural log of acceptance probability of transiting from X to Y'''
        return self.log_prior+self.log_likelihood(X,Y)

    def log_transition(self, target, q=0.5):
        '''Transition probability from self to target'''
        #return target.log_prior()-self.log_prior()+np.log(q)
        # For now assume simple transition model
        return np.log(q)

    def __repr__(self):
        return f'NN({self.num_nodes}, l={self.l}, lr={self.lr}, x_in={self.x_in})'

    def predict(self, X):
        return np.squeeze(self.model.predict(X)) # so sklearn and TF have same shape

if HAVE_TF:
    class TF_NN(NN):
        '''
        Neural Network with single hidden layer implemented with TensorFlow.

        Inherits methods to do MCMC transitions and calculations.
        '''
        def __init__(self,
                num_nodes=10,
                weight_donor=None,
                l=10,
                lr=0.01,
                r=42,
                epochs=20,
                x_in=None,
                batch_size=512,
                solver=None,
                reg=REG,
                act=ACT,
                tol=None):
            assert x_in is not None
            if act == 'logistic':
                act = 'sigmoid'
            self.num_nodes = num_nodes
            tf.keras.utils.set_random_seed(r) #TODO: make sure to vary when calling
            # make an NN with a single hidden layer with num_nodes nodes
            ## can set max_iter to set max_epochs
            self.model = tf.keras.Sequential()
            self.model.add(tkl.Input(shape=(x_in,)))
            self.model.add(tkl.Dense(num_nodes, # hidden layer
                activation=act,
                kernel_regularizer=tkr.L1L2(reg),
                bias_regularizer=tkr.L1L2(reg)))
            self.model.add(tkl.Dense(1,
                kernel_regularizer=tkr.L1L2(reg),
                bias_regularizer=tkr.L1L2(reg))) # output
            # l is poisson shape param, expected number of nodes
            self.l = l
            self.lr = lr
            self.r = r
            self.reg = reg
            self.act = act
            self.epochs = epochs
            self.x_in = x_in
            self.batch_size = batch_size
            if weight_donor is not None:
                # inherit the first num_nodes weights from this donor
                donor_num_nodes = weight_donor.num_nodes
                donor_weights, donor_intercepts = weight_donor.get_weights()
                self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)

        def get_weights(self):
            W = self.model.get_weights()
            # split up so we can handle inheritance
            weights = W[::2]
            intercepts = W[1::2]
            return weights, intercepts

        def accept_donation(self, donor_num_nodes, donor_weights, donor_intercepts):
            '''
            Replace our weights with those of another `NN` (passed as weights).

            Donor can be different size; if smaller, earlier weights in donee
            are overwritten.
            '''
            # a big of a workaround to create weight arrays and things
            num_nodes = self.num_nodes
            #self.model._random_state = crs(self.r)
            #self.model._initialize(np.zeros((1,1),dtype=donor_weights[0].dtype),
            #                       [donor_weights[0].shape[0], num_nodes, 1],
            #                       donor_weights[0].dtype)
            # TODO: generalize this
            if donor_num_nodes == num_nodes:
                self.model.coefs_ = [d.copy() for d in donor_weights]
                self.model.intercepts_ = [d.copy() for d in donor_intercepts]
            elif donor_num_nodes > num_nodes:
                self.model.coefs_ = [donor_weights[0][:,:num_nodes].copy(),
                                     donor_weights[1][:num_nodes].copy()]
                self.model.intercepts_ = [donor_intercepts[0][:num_nodes].copy(),
                                     donor_intercepts[1].copy()]
            else:
                self.model.coefs_, self.model.intercepts_ = self.get_weights()
                self.model.coefs_[0][:,:donor_num_nodes] = donor_weights[0].copy()
                self.model.coefs_[1][:donor_num_nodes] = donor_weights[1].copy()
                self.model.intercepts_[0][:donor_num_nodes] = donor_intercepts[0].copy()
                self.model.intercepts_[1] = donor_intercepts[1].copy()

            W = self.model.coefs_ + self.model.intercepts_
            W[::2] = self.model.coefs_
            W[1::2] = self.model.intercepts_
            # Alternative:
            # import itertools
            # W = list(itertools.chain(*zip(self.model.coefs_, self.model.intercepts_)))
            self.model.set_weights(W)

        def train(self, X, Y):
            '''Train network from current position with given data'''
            self.opt = tf.keras.optimizers.RMSprop(self.lr)
            self.model.compile(self.opt, loss='mse')
            self.model.fit(X,Y, epochs=self.epochs, batch_size=self.batch_size)
else:
    warnings.warn('Unable to use Tensorflow, only sklearn backend available')
    TF_NN = NN


# total acceptable of moving from N to Np given data XY
# TODO: double check this
def A(Np, N, X, Y, sigma, q=0.5):
    '''
    Acceptance ratio of moving from `N` to `Np` given data and 
    transition probability `q` and current sigma est, `sigma`

    Only allows for 2 moves in state transition, grow/shrink
    '''
    # disallow empty network...or does this mean kill it off entirely?
    if Np.num_nodes < 1:
        return 0
    num = Np.log_transition(N,q) + Np.log_likelihood(X, Y, sigma) + Np.log_prior()
    denom = N.log_transition(Np,1-q) + N.log_likelihood(X, Y, sigma) + N.log_prior()
    # assumes only 2 inverse types of transition
    return min(1, np.exp(num-denom))

class BARN_base(BaseEstimator):
    '''
    Bayesian Additive Regression Networks ensemble.
    
    Specify and train an array of neural nets with Bayesian posterior.

    Argument Descriptions:

    * `act` - Activation function, 'logistic' or 'relu', passed to `NN`
    * `batch_size` - batch size for gradient descent solvers, passed to `NN`
    * `burn` - number of MCMC iterations to initially discard for burn-in
    * `callbacks` - dictionary of callbacks, keyed by function with values being arguments passed to the callback
    * `dname` - dataset name if saving output
    * `epochs` - number of neural network training epochs for gradient descent solver
    * `init_neurons` - number of neurons to initialize each network in the ensemble to
    * `l` - prior distribution (Poisson) expected number of neurons for network in ensemble
    * `lr` - learning rate for solvers
    * `n_features_in_` - number of features expected in input data, can be set during `setup_nets` instead
    * `n_iter` - maximum number of MCMC iterations to perform.  One iter is a pass through each network in the ensemble
    * `ndraw` - number of MCMC draws over which to average models
    * `normalize` - flag to normalize input and output before training and prediction
    * `nu` - chi-squared parameter for prior on model error, recommend leaving default
    * `num_nets` - number of networks in the ensemble
    * `qq` - quantile for error prior to compute lambda, recommend leaving default
    * `random_state` - random seed for reproducible results
    * `reg` - L1L2 regularization weight penalty on NN weights, passed to `NN`
    * `silence_warnings` - flag to turn off convergence warnings from SciPy which may not be helpful
    * `solver` - solver preference ('lbfgs' or 'adam'), use `None` to automatically select based on problem size, passed to `NN`
    * `test_size` - fraction of data to use as held-back testing data if not supplying separate splits to `BARN.fit`
    * `tol` - tolerance used in solver, passed to `NN`
    * `trans_options` - possible state transition options as a list of strings
    * `trans_probs` - probability of each state transition option, currently fixed for each option
    * `use_tf` - flag to use TensorFlow backend for `NN`, recommend leaving `False` unless networks are large
    * `warm_start` - flag to allow reusing previously trained ensemble for a given instance (`True`) or create a fresh ensemble with calling `BARN.fit` (`False`), irrelevant if only calling `BARN.fit` for an instance once
    '''
    def __init__(self,
            act=ACT,
            batch_size=512,
            burn=None,
            callbacks=dict(),
            dname='default_name',
            epochs=10,
            init_neurons=None,
            l=2,
            lr=0.01,
            n_features_in_=None,
            n_iter=200,
            ndraw=None,
            normalize=True,
            nu=3,
            num_nets=10,
            qq=0.9,
            random_state=42,
            reg=REG,
            silence_warnings=True,
            solver=None,
            test_size=0.25,
            tol=1e-3,
            trans_options=['grow', 'shrink'],
            trans_probs=[0.4, 0.6],
            use_tf=False,
            verbose=False,
            warm_start=True,
            ):
        self.num_nets = num_nets
        # check that transition probabilities look like list of numbers
        try:
            np.sum(trans_probs)
        except TypeError as e:
            raise TypeError(f"trans_probs needs to be a list of numbers, not {trans_probs}") from e
        # maybe should bias to shrinking to avoid just overfitting?
        # or compute acceptance resid on validation data?
        try:
            assert len(trans_probs) == len(trans_options)
        except AssertionError as e:
            raise IndexError(f"Number of probabilities in trans_probs ({len(trans_probs)}) needs to equal number of options in trans_options ({len(trans_options)})")
        self.trans_probs = trans_probs
        self.trans_options = trans_options
        self.dname = dname
        self.random_state = random_state
        self.np_random_state = np.random.RandomState(seed=random_state)
        use_tf = use_tf & HAVE_TF
        if use_tf:
            self.NN = TF_NN
        else:
            self.NN = NN
        self.use_tf = use_tf
        self.batch_size = batch_size
        self.solver = solver
        self.l = l
        self.lr = lr
        self.epochs = epochs
        self.n_iter = n_iter
        self.test_size = test_size
        self.initialized=False
        self.warm_start=warm_start
        self.n_features_in_ = n_features_in_
        self.tol = tol
        self.callbacks = callbacks
        self.reg = reg
        self.act = act
        self.nu = nu
        self.qq = qq
        self._binary = False # private option for internal testing, *does not* make BARN binary classifier (use BARN_bin for that)
        if init_neurons is None:
            self.init_neurons = 1
        else:
            self.init_neurons = init_neurons
        self.verbose = verbose

        self.silence_warnings = silence_warnings
        if silence_warnings:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

        if burn:
            self.burn = burn
        else:
            self.burn = self.n_iter // 2
        # TODO: compute batch means or autocorrelation on the fly to estimate number of samples to save
        if ndraw:
            self.save_every = min(self.n_iter - self.burn, max(1,(self.n_iter - self.burn)//ndraw))
        else:
            self.save_every = None
        self.ndraw = ndraw # unused
        self.normalize = normalize
        self.scale_x = None
        self.scale_y = None

    def setup_nets(self, n_features_in_=None):
        '''
        Intermediate method to initialize networks in ensemble
        '''
        if n_features_in_ is None:
            n_features_in_ = self.n_features_in_
        elif self.n_features_in_ is None:
            self.n_features_in_ = n_features_in_
        self.cyberspace = [self.NN(self.init_neurons, l=self.l, lr=self.lr, epochs=self.epochs, r=self.random_state+i, x_in=n_features_in_, batch_size=self.batch_size, solver=self.solver, tol=self.tol, reg=self.reg, act=self.act, binary=self._binary) for i in range(self.num_nets)]
        self.initialized=True
        self.saved_draws = []
        self.sigma_draws = []

    def sample_sigma(self):
        '''
        Sample model sigma from posterior distribution (another inverse gamma)
        '''
        n = self.Xtr.shape[0]
        preds = np.sum([N.predict(self.Xtr) for N in self.cyberspace], axis=0)
        sse = np.sum((self.Ytr-preds)**2)
        post_alpha = self.prior_alpha + n/2
        post_beta = 2/(2/self.prior_beta + sse)
        return np.sqrt(scipy.stats.invgamma.rvs(post_alpha, scale=1/post_beta,
                    random_state=self.np_random_state))
        #dof = (self.nu + n) # good, based on invchi2 origin
        #tau_sq = (self.nu * self.sigma0+nu1)/dof
        #a =  dof/2
        #s = dof*tau_sq/2
        #return scipy.stats.invgamma.rvs(a, scale=s)

    def compute_res(self, X, Y, i, S=None):
        '''
        Compute the residual for the current iteration, returning total prediction as well as target without contribution from model i.

        Optionally use an existing S from previous iteration
        '''
        if S is None:
            S = np.array([N.predict(X) for N in self.cyberspace])
        R = Y - (np.sum(S, axis=0) - S[i])
        return S, R

    def update_target(self, X, Y_old):
        return Y_old

    def fit(self, X, Y, Xva=None, Yva=None, Xte=None, Yte=None, n_iter=None):
        '''
        Overall BARN fitting method.

        If `Xva`/`Yva` not supplied yet such data is requested by `self.test_size`,
        then training data is split, using `self.test_size` fraction as validation.
        If this validation data is available, it's used for acceptance.  Otherwise,
        the training data is reused for the probability calculation.

        If `Xte`/`Yte` not supplied, however, we skip the test data calculation.
        '''
        if not self.initialized or not self.warm_start:
            self.n_features_in_ = X.shape[1]
            self.setup_nets()
        Xtr = X
        Ytr = Y
        if n_iter is None:
            n_iter = self.n_iter
        else:
            self.n_iter = n_iter # ok to overwrite?
            pass

        if Xva is None and self.test_size > 0:
            Xtr, XX, Ytr, YY = skms.train_test_split(Xtr,Ytr,
                                test_size=self.test_size,
                                random_state=self.random_state) # training
            #if Xte is None:
            #    Xva, Xte, Yva, Yte = skms.train_test_split(XX,YY,
            #            test_size=self.test_size,
            #            random_state=self.random_state) # valid and test
            #else:
            Xva = XX
            Yva = YY

        # optionally normalize data
        self.scale_x = self.create_input_normalizer(Xtr, self.normalize)
        self.scale_y = self.create_output_normalizer(Ytr, self.normalize)
        if self.normalize:
            Xtr = self.scale_x.transform(Xtr)
            Ytr = self.scale_y.transform(Ytr.reshape((-1,1))).reshape(-1)
            if Xva is not None:
                Xva = self.scale_x.transform(Xva)
                Yva = self.scale_y.transform(Yva.reshape((-1,1))).reshape(-1)


        # initialize fit as though all get equal share of Y
        for j,N in enumerate(self.cyberspace):
            N.train(Xtr,Ytr/self.num_nets)

        # check initial fit
        Yh_tr = np.sum([N.predict(Xtr) for N in self.cyberspace], axis=0)
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Ytr_init = np.copy(Yh_tr)
        if Xte is not None:
            Xte = scale_x.transform(Xte)
            Yh = np.sum([N.predict(Xte) for N in self.cyberspace], axis=0)
            self.Yte_init = np.copy(Yh)

        # Compute initial sigma guess from simple Y var for now, OLS fit later
        ## this is intended to be an overestimate
        sigma_hat = np.std(Ytr)
        ff = lambda bb: (self.qq-scipy.stats.invgamma.cdf(sigma_hat**2, self.nu/2, scale=bb))**2
        #t2 = scipy.optimize.bisect(ff, 0, 100)
        self.prior_beta = 1/scipy.optimize.minimize(ff, 1, method='nelder-mead').x # doesn't need to be that close
        self.prior_alpha = self.nu/2
        #self.prior_beta = 2/(self.nu*self.t2)
        # return StatToolbox.sample_from_inv_gamma((hyper_nu + es.length) / 2, 2 / (sse + hyper_nu * hyper_lambda)); from bartmachine
        #self.sigma0 = scipy.stats.invgamma.rvs(self.nu/2, scale=self.nu*self.t2/2)

        # initial sigma sample
        self.sigma = self.sample_sigma()

        self.accepted = 0
        self.phi = np.zeros(n_iter)
        self.sigmas = np.zeros(n_iter)
        self.ntrans_iter = np.zeros(n_iter)
        self.actual_num_neuron = np.zeros((self.n_iter,
                                           self.num_nets),
                                           dtype=np.uint8)

        Ytr_orig = np.copy(Ytr)
        if Yva is not None:
            Yva_orig = np.copy(Yva)
        else:
            Yva_orig = None

        if n_iter > 0:
            Ytr = self.update_target(Xtr, Ytr_orig)
            if Yva is not None:
                Yva = self.update_target(Xva, Yva_orig)
            # setup residual array
            S_tr, Rtr = self.compute_res(Xtr, Ytr, -1, None)
            if Xva is not None:
                S_va, Rva = self.compute_res(Xva, Yva, -1, None)
            try:
                for i in tqdm(range(n_iter)):
                    self.i = i # only for checking callbacks
                    for callback,kwargs in self.callbacks.items():
                        res = callback(self, **kwargs)

                    if i > 0: #TODO bin
                    #if False:
                        # latent Z sampling for binary class (ignored for regression)
                        Ytr = self.update_target(Xtr, Ytr_orig)
                        if Yva is not None:
                            Yva = self.update_target(Xva, Yva_orig)
                        S_tr, Rtr = self.compute_res(Xtr, Ytr, -1, S_tr)
                        if Xva is not None:
                            S_va, Rva = self.compute_res(Xva, Yva, -1, S_va)

                    # gibbs sample over the nets
                    for j in range(self.num_nets):
                        # compute resid against other nets
                        ## Use cached these results, add back most recent and remove current
                        ## TODO: double check this is correct
                        if Xva is not None:
                            Rva = Rva - S_va[j-1] + S_va[j]
                        Rtr = Rtr - S_tr[j-1] + S_tr[j]

                        # grab current net in this position
                        N = self.cyberspace[j]
                        # create proposed change
                        choice = self.np_random_state.choice(self.trans_options, p=self.trans_probs)
                        if choice == 'grow':
                            Np = self.NN(N.num_nodes+1, weight_donor=N, l=N.l, lr=N.lr, r=self.np_random_state.randint(BIG), x_in=self.n_features_in_, epochs=self.epochs, batch_size=self.batch_size, solver=self.solver, tol=self.tol, reg=self.reg, act=self.act)
                            q = self.trans_probs[0]
                        elif N.num_nodes-1 == 0:
                            # TODO: better handle zero neuron case, don't just skip
                            continue # don't bother building empty model
                        else:
                            Np = self.NN(N.num_nodes-1, weight_donor=N, l=N.l, lr=N.lr, r=self.np_random_state.randint(BIG), x_in=self.n_features_in_, epochs=self.epochs, solver=self.solver, tol=self.tol, reg=self.reg, act=self.act)
                            q = self.trans_probs[1]
                        Np.train(Xtr,Rtr)
                        # determine if we should keep it
                        if Xva is not None:
                            Xcomp = Xva
                            Rcomp = Rva
                        else:
                            Xcomp = Xtr
                            Rcomp = Rtr
                        if self.np_random_state.random() < A(Np, N, Xcomp, Rcomp, self.sigma, q):
                            self.cyberspace[j] = Np
                            self.accepted += 1
                            self.ntrans_iter[i] += 1
                            S_tr[j] = Np.predict(Xtr)
                            if Xva is not None:
                                S_va[j] = Np.predict(Xva)
                    if Xva is not None:
                        Rphi = Rva
                        S_phi = S_va
                    else:
                        Rphi = Rtr
                        S_phi = S_tr
                    # overall validation error at this MCMC iteration
                    self.phi[i] = np.sqrt(np.mean((Rphi - S_phi[j])**2))
                    self.actual_num_neuron[i] = [m.num_nodes for m in self.cyberspace]
                    self.sigma = self.sample_sigma()
                    self.sigmas[i] = self.sigma
                    # save multiple draws for averaging in prediction
                    self.multidraw()

            except JackPot:
                # indicates we ended early
                self.n_iter = i-1
                # shorten the saved results (or fill rest with NaNs?
                self.phi = self.phi[:i-1]
                self.ntrans_iter = self.ntrans_iter[:i-1]
                self.multidraw()

        if len(self.saved_draws) == 0:
            self.saved_draws.append(self.cyberspace)
        if Xte is not None:
            self.Xte = Xte
            self.Yte = Yte
        return self

    def predict_all(self, X):
        '''
        Return all drawn MCMC predictions for given input data
        '''
        X_tmp = self.scale_x.transform(X)
        ans = np.array([np.sum([N.predict(X_tmp) for N in cyberspace], axis=0) for cyberspace in self.saved_draws])
        sh = (len(self.saved_draws), X_tmp.shape[0])
        return self.scale_y.inverse_transform(ans.reshape((-1,1))).reshape(sh)

    def predict(self, X):
        return np.mean(self.predict_all(X), axis=0)

    def predict_interval(self, X, alpha=0.05):
        '''
        Predict upper and lower confidence interval at alpha level
        Aggregate multiple MCMC draws using t-stat of point est
        '''
        all_ans = self.predict_all(X)
        intervals = np.zeros((all_ans.shape[1],2))
        # No `axis` command for hdi, loop through
        for i in range(all_ans.shape[1]):
            intervals[i] = pymc.stats.hdi(all_ans[:,i], 1-alpha)
        return intervals[:,0], intervals[:,1]

    def phi_viz(self, outname='phi.png', close=True):
        '''
        Visualize the `phi` parameter, validation error over time
        '''
        fig = plt.figure()
        fig.set_size_inches(5,5)
        # Plot phi results
        plt.plot(self.phi)
        plt.xlabel('MCMC Iteration')
        plt.ylabel('RMSE')
        plt.title(f'MCMC Error Progression')
        if outname:
            fig.savefig(outname)
        if close:
            plt.close()
        return fig

    def viz(self, outname='results.png', extra_slots=0, close=True, initial=False, do_viz=True):
        '''
        Visualize results of BARN analysis.

        Shows:
            1. Plot of `Y_test` vs `Y_test_pred_initial` (optional)
            2. Plot of `Y_test` vs `Y_test_pred_final`

        Requires existing `self.Xte` and `self.Yte` (usually set by `self.fit`)
        '''
        fig, ax = plt.subplots(1,1+extra_slots+initial, squeeze=True, sharex=True,sharey=True)
        fig.set_size_inches(12+4*extra_slots,4)
        if initial:
            # check initial fit
            # replace r2 with training r2
            #r2h = metrics.r2_score(self.Yte, self.Yte_init)
            r2h = metrics.r2_score(self.Ytr, self.Ytr_init)
            rmseh = metrics.mean_squared_error(self.Yte, self.Yte_init, squared=False)

            ax[0].plot([np.min(self.Yte), np.max(self.Yte)],
                       [np.min(self.Yte), np.max(self.Yte)])
            ax[0].scatter(self.Yte,self.Yte_init, c='orange') # somewhat decent on synth, gets lousy at edge, which makes sense
            ax[0].set_title('Initial BARN')
            ax[0].set_ylabel('Prediction')
            ax[0].text(0.05, 0.85, f'Training $R^2 = $ {r2h:0.4}\nTest $RMSE = $ {rmseh:0.4}', transform=ax[0].transAxes)
        elif extra_slots + initial == 0:
            # pretend to have a list so we can access by index
            ax = [ax]
        else:
            # should be ok
            pass

        # final fit
        Yh2 = self.predict(self.Xte)
        #r2h2 = metrics.r2_score(self.Yte, Yh2)
        r2h2 = metrics.r2_score(self.Ytr, self.predict(self.Xtr))
        rmseh2 = metrics.mean_squared_error(self.Yte, Yh2, squared=False)
        ax[0+initial].plot([np.min(self.Yte), np.max(self.Yte)],
                   [np.min(self.Yte), np.max(self.Yte)])
        ax[0+initial].scatter(self.Yte,Yh2, c='orange')
        ax[0+initial].set_title('Final BARN')
        ax[0+initial].set_xlabel('Target')
        ax[0+initial].text(0.05, 0.85, f'Training $R^2 = $ {r2h2:0.4}\nTest $RMSE = $ {rmseh2:0.4}', transform=ax[0+initial].transAxes)

        if do_viz:
            fig.savefig(outname)
        if close:
            plt.close()
        return fig, ax, rmseh2, r2h2

    def batch_means(self, num_batch=20, batch_size=None, np_out='val_resid.npy', outfile='var_all.csv', mode='a', burn=None, num=None):
        '''
        Compute batch means variance over computed results.
        '''
        if burn is None:
            burn = 100
        if batch_size is None:
            batch_size = self.total_iters//num_batch
        if num is None:
            num = min(len(self.phi[burn:]), num_batch*batch_size)
        if num//num_batch != batch_size:
            num -= num % batch_size
            num_batch = int(num//batch_size)
        # check batch means variance
        mu = np.mean(self.phi[burn:burn+num])
        if np_out:
            np.save(np_out, self.phi) # only final saved
        batch_phi = np.mean(self.phi[burn:burn+num].reshape((num_batch, batch_size)), axis=1)
        var = np.sum((batch_phi-mu)**2)/(num_batch*(num_batch-1))
        if outfile:
            with open(outfile, mode) as f:
                print(f'{self.dname}, {var}', file=f)
        return var

    def save(self, outname):
        '''
        Save the entire ensemble of NNs as a Python pickle.  Load with pickle too.
        '''
        # This only saves the last iteration of full models, but that's something
        if self.use_tf:
            # convert to sklearn and save? Don't actually have a load method yet!
            # could reconstruct from just weights if needed
            with open(outname,'wb') as f:
                pickle.dump(self.get_weights(), f)
        else:
            with open(outname,'wb') as f:
                pickle.dump(self.cyberspace, f)

    def get_weights(self):
        '''
        Obtain weights of the NNs in the ensemble.
        '''
        return [nn.get_weights() for nn in self.cyberspace]

    @staticmethod
    def improvement(self, check_every=None, skip_first=0, tol=0):
        '''
        Stop early if performance has not improved for `check_every` iters.

        Allow wiggle room such that if we are within `tol` % of old best, continue

        Skip the first `skip_first` iters without checking
        '''
        i = self.i
        if check_every is None:
            check_every = max(self.n_iter//10, 1)
        # not an iteration to stop on
        if i == 0 or i % check_every != 0:
            return None
        if i < skip_first or i-check_every <= 0:
            return None
        tol = 1+tol
        old_best = np.min(self.phi[:i-check_every])
        recent_best = np.min(self.phi[i-check_every:i])
        # if it's getting worse, then stop
        print(old_best*tol, recent_best)
        if old_best*tol <= recent_best:
            print(f'Ended early on {i}!')
            raise JackPot
        else:
            return None

    @staticmethod
    def trans_enough(self, check_every=None, skip_first=0, ntrans=None):
        '''
        Stop early if fewer than `ntrans` transitions

        Skip the first `skip_first` iters without checking
        '''
        i = self.i
        if check_every is None:
            check_every = max(self.n_iter//10, 1)
        # not an iteration to stop on
        if i == 0 or i % check_every != 0:
            return None
        if i < skip_first:
            return None
        if ntrans is None:
            ntrans = max(self.num_nets//5,1)
        if self.ntrans_iter[i-1] < ntrans: # maybe check mean percent across block?
            raise JackPot

    @staticmethod
    def stable_dist(self, check_every=None, skip_first=0, tol=None):
        '''
        Stop early if posterior distribution of neuron distribution is stable

        Tolerance is on Wassersten metric (aka Earth Mover Distance)

        Skip the first `skip_first` iters without checking
        '''
        i = self.i
        if check_every is None:
            check_every = max(self.n_iter//10, 1)
        if tol is None:
            tol = self.num_nets//10 # 10% of nets can change one value
        # not an iteration to stop on
        if i == 0 or i % check_every != 0:
            return None
        if i < skip_first:
            return None
        # find max Earth Mover Distance in last successive check_every steps
        max_dist = None
        for j in range(check_every):
            idx = i-check_every+j-1
            # skip if trying to compare before first entry
            if idx < 0:
                continue
            ws = scipy.stats.wasserstein_distance(
                    self.actual_num_neuron[idx],
                    self.actual_num_neuron[idx+1])
            if max_dist is None or ws > max_dist:
                max_dist = ws
        if max_dist <= tol:
            raise JackPot

    @staticmethod
    def rfwsr(self, check_every=None, skip_first=0, t=2, eps=0.01):
        '''
        Relative Fixed Width Stopping Rule

        `t*sig/sqrt(n) + 1/n <= eps * gbar`

        Skip the first `skip_first` iters without checking
        '''
        i = self.i
        if check_every is None:
            check_every = max(self.n_iter//10, 1)
        # not an iteration to stop on
        if i == 0 or i % check_every != 0:
            return None
        if i < skip_first:
            return None
        # only use last half of steps, to avoid burn-in period
        num_batch = i//(2*check_every)
        if num_batch == 0:
            return None
        num = num_batch*check_every
        burn = i-num
        gbar = np.mean(self.phi[burn:i]) 
        sig = self.batch_means(num_batch=num_batch,
                batch_size=check_every,
                num=num,
                np_out='', outfile='', burn=burn)
        if t*sig/np.sqrt(num)<= eps*gbar: # removed 1/n term on LHS
            raise JackPot

    def multidraw(self):
        '''
        Save multiple draws of the multiple for averaged prediction

        Skip the first `self.burn` iters
        Otherwise, keep every `self.save_every` model
        '''
        i = self.i
        if self.save_every is None:
            self.save_every = 10 # default 10%
        # not an iteration to stop on
        if i > 0 and i >= self.burn and i % self.save_every == 0:
            self.saved_draws.append(self.cyberspace[:])
            self.sigma_draws.append(self.sigma)

    def create_input_normalizer(self, X, normalize=False):
        '''
        Setup PCA normalization of input features
        '''
        if normalize:
            scale_x = PCA(n_components=X.shape[1], whiten=False)
        else:
            # dummy scaler
            scale_x = StandardScaler(with_mean=False, with_std=False)
        scale_x.fit(X)
        return scale_x
        
    def create_output_normalizer(self, Y, normalize=False):
        '''
        Setup PCA normalization of target output
        '''
        scale_y = StandardScaler(with_mean=normalize, with_std=normalize)
        scale_y.fit(Y.reshape((-1,1)))
        return scale_y

class BARN(BARN_base, RegressorMixin):
    pass

class BARN_bin(BARN_base, ClassifierMixin):
    '''
    Bayesian Additive Regression Networks ensemble for binary classification.
    '''
    def create_output_normalizer(self, Y, normalize=False):
        # leave classes untouched
        scale_y = StandardScaler(with_mean=False, with_std=False)
        scale_y.fit(Y.reshape((-1,1)))
        return scale_y

    def sample_sigma(self):
        return 1

    def predict(self, X):
        return scipy.stats.norm.cdf(self.predict_z(X))

    def predict_z_all(self, X):
        X_tmp = self.scale_x.transform(X)
        return np.array([np.sum([N.predict(X_tmp).reshape(-1) for N in cyberspace], axis=0) for cyberspace in self.saved_draws])

    def predict_all(self, X):
        return scipy.stats.norm.cdf(self.predict_z_all(X))

    def predict_z(self, X):
        '''
        Return prediction as z-score
        '''
        if len(self.saved_draws) > 0:
            ans = np.mean(self.predict_z_all(X), axis=0)
        else:
            X_tmp = self.scale_x.transform(X)
            ans = np.sum([N.predict(X_tmp) for N in self.cyberspace], axis=0)
        return self.scale_y.inverse_transform(ans.reshape((-1,1))).reshape(-1)

    def predict_z_interval(self, X, alpha=0.05):
        all_ans = self.predict_z_all(X)
        # old CI was on mean, not posterior
        #z = scipy.stats.norm.ppf(1-alpha/2)
        #ans = self.predict_z(X)
        #n = len(self.sigma_draws)
        #sig_unscaled = n # fixed sigma_i=1 for classification
        #return ans - z*sig_unscaled/np.sqrt(n), ans + z*sig_unscaled/np.sqrt(n)
        intervals = np.zeros((all_ans.shape[1],2))
        # No `axis` command for hdi, loop through
        for i in range(all_ans.shape[1]):
            intervals[i] = pymc.stats.hdi(all_ans[:,i], 1-alpha)
        return intervals[:,0], intervals[:,1]


    def predict_interval(self, X, alpha=0.05):
        lower, upper = self.predict_z_interval(X)
        return scipy.stats.norm.cdf(lower), scipy.stats.norm.cdf(upper)

    def update_target(self, X, Y_old):
        '''
        Estimate latent $z_i$ values with current model and data.
        
        Essentially, make prediction of z-score with current model and
        clip to zero if sign is wrong.

        $z_i ~ max(N(g(X),1), 0) y_i + min(N(g(X),1), 0) (1-y_i) $
        '''

        pz = self.predict_z(X)
        sample = scipy.stats.norm.rvs(loc=pz, random_state=self.np_random_state)
        return np.clip(sample*Y_old,0, None) + np.clip(sample*(1-np.array(Y_old)),None, 0)



# A way to break out of nested for loops without trying to be clever with flags
class JackPot(Exception):
    pass
