try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# use sklearn for now, could upgrade to keras later if we want
import sklearn.neural_network as sknn
from sklearn.utils import check_random_state as crs
import sklearn.model_selection as skms
import sklearn.metrics as metrics
import pickle
import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.regularizers as tkr

INFO = np.iinfo(np.int32)
SMALL = INFO.min + 1
BIG = INFO.max - 1
REG = 0.001 # regularization

class NN(object):
    '''
    Neural Network with single hidden layer implemented with sklearn.

    Includes methods to do MCMC transitions and calculations.
    '''
    def __init__(self,
            num_nodes=10,
            weight_donor=None,
            l=10,
            lr=0.01,
            r=42,
            epochs=20,
            x_in=None,
            batch_size=512):
        self.num_nodes = num_nodes
        self.x_in = x_in
        # make an NN with a single hidden layer with num_nodes nodes
        ## can set max_iter to set max_epochs
        if num_nodes < 10:
            solver = 'lbfgs'
        else:
            solver = 'adam'
        self.model = sknn.MLPRegressor([num_nodes],
                learning_rate_init=lr,
                random_state=r,
                max_iter=epochs,
                batch_size=batch_size,
                warm_start=True,
                solver=solver)
        # l is poisson shape param, expected number of nodes
        self.l = l
        self.lr = lr
        if r is None:
            r = int(time.time())
        self.r = r
        self.epochs = epochs
        self.x_in = x_in
        if weight_donor is not None:
            # inherit the first num_nodes weights from this donor
            donor_num_nodes = weight_donor.num_nodes
            donor_weights, donor_intercepts = weight_donor.get_weights()
            self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)

    def save(self, fname):
        params = np.array([self.num_nodes, self.l, self.lr, self.r,
            self.epochs, self.x_in])
        coefs_, intercepts_ = self.model.get_weights()
        np.savez_compressed(fname, params=params,
                coefs_=coefs_,
                intercepts_=intercepts_)

    def get_weights(self):
        return (self.model.coefs_, self.model.intercepts_)

    def accept_donation(self, donor_num_nodes, donor_weights, donor_intercepts):
        '''
        Replace our weights with those of another `NN` (passed as weights).

        Donor can be different size; if smaller, earlier weights in donee
        are overwritten.
        '''
        # a big of a workaround to create weight arrays and things
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
        network = np.load(fname)
        N = NN(network['params'][0],
               l=network['params'][1],
               lr=network['params'][2],
               r=network['params'][3],
               epochs=network['params'][4],
               x_in=network['params'][5]
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

    def log_likelihood(self, X, Y):
        '''
        Log likelihood of `NN` assuming normally distributed errors.
        '''
        # compute residuals
        yhat = np.squeeze(self.model.predict(X))
        resid = Y - yhat
        # compute stddev of these
        std = np.std(resid) # maybe use std prior here?
        # normal likelihood
        return np.sum(scipy.stats.norm.logpdf(resid, 0, std))

    def log_acceptance(self, X, Y):
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
            batch_size=512):
        self.num_nodes = num_nodes
        tf.keras.utils.set_random_seed(r) #TODO: make sure to vary when calling
        # make an NN with a single hidden layer with num_nodes nodes
        ## can set max_iter to set max_epochs
        self.model = tf.keras.Sequential()
        self.model.add(tkl.Input(shape=(x_in,)))
        self.model.add(tkl.Dense(num_nodes, # hidden layer
            activation='relu',
            kernel_regularizer=tkr.L1L2(REG),
            bias_regularizer=tkr.L1L2(REG)))
        self.model.add(tkl.Dense(1,
            kernel_regularizer=tkr.L1L2(REG),
            bias_regularizer=tkr.L1L2(REG))) # output
        # l is poisson shape param, expected number of nodes
        self.l = l
        self.lr = lr
        self.r = r
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


# total acceptable of moving from N to Np given data XY
# TODO: double check this
def A(Np, N, X, Y, q=0.5):
    '''
    Acceptance ratio of moving from `N` to `Np` given data and 
    transition probability `q`

    Only allows for 2 moves in state transition, grow/shrink
    '''
    # disallow empty network...or does this mean kill it off entirely?
    if Np.num_nodes < 1:
        return 0
    num = Np.log_transition(N,q) + Np.log_likelihood(X, Y) + Np.log_prior()
    denom = N.log_transition(Np,1-q) + N.log_likelihood(X, Y) + N.log_prior()
    # assumes only 2 inverse types of transition
    return min(1, np.exp(num-denom))

class BARN(object):
    '''
    Bayesian Additive Regression Networks ensemble.
    
    Specify and train an array of neural nets with Bayesian posterior.
    '''
    def __init__(self, num_nets=10,
            trans_probs=[0.4, 0.6],
            trans_options=['grow', 'shrink'],
            dname='default_name',
            random_state=42,
            x_in=None,
            use_tf=False,
            batch_size=512):
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
        assert x_in is not None
        self.x_in = x_in
        if use_tf:
            self.NN = TF_NN
        else:
            self.NN = NN
        self.use_tf = use_tf
        self.batch_size = batch_size

    def setup_nets(self, l=10, lr=0.01, epochs=10):
        self.epochs = epochs
        self.cyberspace = [self.NN(1, l=l, lr=lr, epochs=epochs, r=self.random_state+i, x_in=self.x_in, batch_size=self.batch_size) for i in range(self.num_nets)]

    def train(self, Xtr, Ytr, Xva=None, Yva=None, Xte=None, Yte=None, total_iters=10):
        if Xva is None:
            Xtr, XX, Ytr, YY = skms.train_test_split(Xtr,Ytr,
                                test_size=0.5,
                                random_state=self.random_state) # training
            if Xte is None:
                Xva, Xte, Yva, Yte = skms.train_test_split(XX,YY,
                        test_size=0.5,
                        random_state=self.random_state) # valid and test
            else:
                Xva = XX
                Yva = YY

        # initialize fit as though all get equal share of Y
        for j,N in enumerate(self.cyberspace):
            N.train(Xtr,Ytr/self.num_nets)

        # check initial fit
        Yh = np.sum([N.predict(Xte) for N in self.cyberspace], axis=0)
        self.Yte_init = np.copy(Yh)

        accepted = 0
        # setup residual array
        S_tr = np.array([N.predict(Xtr) for N in self.cyberspace])
        S_va = np.array([N.predict(Xva) for N in self.cyberspace])
        Rtr = Ytr - (np.sum(S_tr, axis=0) - S_tr[-1])
        Rva = Yva - (np.sum(S_va, axis=0) - S_va[-1])
        phi = np.zeros(total_iters)
        for i in tqdm(range(total_iters)):
            # gibbs sample over the nets
            for j in range(self.num_nets):
                # compute resid against other nets
                ## Use cached these results, add back most recent and remove current
                ## TODO: double check this is correct
                Rva = Rva - S_va[j-1] + S_va[j]
                Rtr = Rtr - S_tr[j-1] + S_tr[j]

                # grab current net in this position
                N = self.cyberspace[j]
                # create proposed change
                choice = np.random.choice(self.trans_options, p=self.trans_probs)
                if choice == 'grow':
                    Np = self.NN(N.num_nodes+1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG), x_in=self.x_in, epochs=self.epochs, batch_size=self.batch_size)
                    q = self.trans_probs[0]
                elif N.num_nodes-1 == 0:
                    # TODO: better handle zero neuron case, don't just skip
                    continue # don't bother building empty model
                else:
                    Np = self.NN(N.num_nodes-1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG), x_in=self.x_in, epochs=self.epochs)
                    q = self.trans_probs[1]
                Np.train(Xtr,Rtr)
                # determine if we should keep it
                if np.random.random() < A(Np, N, Xva, Rva, q):
                    self.cyberspace[j] = Np
                    accepted += 1
                    S_tr[j] = Np.predict(Xtr)
                    S_va[j] = Np.predict(Xva)
            # overall validation error at this MCMC iteration
            phi[i] = np.sqrt(np.mean((Rva - S_va[j])**2))
        self.phi = phi
        self.accepted = accepted
        self.Xte = Xte
        self.Yte = Yte

    def predict(self, X):
        return np.sum([N.predict(X) for N in self.cyberspace], axis=0)

    def phi_viz(self, outname='phi.png', close=True):
        '''
        Visualize the `phi` parameter, validation error over time
        '''
        fig = plt.figure()
        fig.set_size_inches(4,4)
        # Plot phi results
        plt.plot(self.phi)
        plt.xlabel('MCMC Iteration')
        plt.ylabel('RMSE')
        plt.title(f'MCMC Error Progression')
        fig.savefig(outname)
        if close:
            plt.close()
        return fig

    def viz(self, outname='results.png', extra_slots=0, close=True, initial=False):
        '''
        Visualize results of BARN analysis.

        Shows:
            1. Plot of `Y_test` vs `Y_test_pred_initial` (optional)
            2. Plot of `Y_test` vs `Y_test_pred_final`
        '''
        fig, ax = plt.subplots(1,1+extra_slots+initial, squeeze=True, sharex=True,sharey=True)
        fig.set_size_inches(12+4*extra_slots,4)
        if initial:
            # check initial fit
            r2h = metrics.r2_score(self.Yte, self.Yte_init)
            rmseh = metrics.mean_squared_error(self.Yte, self.Yte_init, squared=False)

            ax[0].plot([np.min(self.Yte), np.max(self.Yte)],
                       [np.min(self.Yte), np.max(self.Yte)])
            ax[0].scatter(self.Yte,self.Yte_init, c='orange') # somewhat decent on synth, gets lousy at edge, which makes sense
            ax[0].set_title('Initial BARN')
            ax[0].set_ylabel('Prediction')
            ax[0].text(0.05, 0.85, f'$R^2 = $ {r2h:0.4}\n$RMSE = $ {rmseh:0.4}', transform=ax[0].transAxes)
        elif extra_slots + initial == 0:
            # pretend to have a list so we can access by index
            ax = [ax]
        else:
            # should be ok
            pass

        # final fit
        Yh2 = self.predict(self.Xte)
        r2h2 = metrics.r2_score(self.Yte, Yh2)
        rmseh2 = metrics.mean_squared_error(self.Yte, Yh2, squared=False)
        ax[0+initial].plot([np.min(self.Yte), np.max(self.Yte)],
                   [np.min(self.Yte), np.max(self.Yte)])
        ax[0+initial].scatter(self.Yte,Yh2, c='orange')
        ax[0+initial].set_title('Final BARN')
        ax[0+initial].set_xlabel('Target')
        ax[0+initial].text(0.05, 0.85, f'$R^2 = $ {r2h2:0.4}\n$RMSE = $ {rmseh2:0.4}', transform=ax[0+initial].transAxes)

        fig.savefig(outname)
        if close:
            plt.close()
        return fig, ax, rmseh2

    def batch_means(self, num_batch=20, batch_size=None, np_out='val_resid.npy', outfile='var_all.csv', mode='a', burn=None):
        '''
        Compute batch means variance over computed results.
        '''
        if burn is None:
            burn = 100
        if batch_size is None:
            batch_size = self.total_iters//num_batch
        # check batch means variance
        mu = np.mean(self.phi[burn:])
        if np_out:
            np.save(np_out, self.phi) # only final saved
        batch_phi = np.mean(self.phi[burn:].reshape((num_batch, batch_size)), axis=1)
        var = np.sum((batch_phi-mu)**2)/(num_batch*(num_batch-1))
        if outfile:
            with open(outfile, mode) as f:
                print(f'{self.dname}, {var}', file=f)
        return var

    def save(self, outname):
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
        return [nn.get_weights() for nn in self.cyberspace]
