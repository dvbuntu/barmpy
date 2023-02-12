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

INFO = np.iinfo(np.int32)
SMALL = INFO.min + 1
BIG = INFO.max - 1

class NN(object):
    '''
    Neural Network with single hidden layer implemented with sklearn.

    Includes methods to do MCMC transitions and calculations.
    '''
    def __init__(self, num_nodes=10, weight_donor=None, l=10, lr=0.01, r=None):
        self.num_nodes = num_nodes
        # make an NN with a single hidden layer with num_nodes nodes
        ## can set max_iter to set max_epochs
        self.model = sknn.MLPRegressor([num_nodes], learning_rate_init=lr, random_state=r)
        # l is poisson shape param, expected number of nodes
        self.l = l
        self.lr = lr
        self.r = r
        if weight_donor is not None:
            # inherit the first num_nodes weights from this donor
            donor_num_nodes = weight_donor.num_nodes
            donor_weights = weight_donor.model.coefs_
            donor_intercepts = weight_donor.model.intercepts_
            self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)

    def save(self, fname):
        params = np.array([self.num_nodes, self.l, self.lr, self.r])
        np.savez_compressed(fname, params=params,
                coefs_=self.model.coefs_,
                intercepts_=self.model.intercepts_)

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
               r=network['params'][3])
        donor_num_nodes = N.num_nodes
        donor_weights = network['coefs_']
        donor_intercepts_ = network['intercepts_']
        self.accept_donation(donor_num_nodes, donor_weights, donor_intercepts)
        return N

    def train(self, X, Y, epochs=10):
        '''Train network from current position with given data'''
        # TODO: figure out how to fix num epochs
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
        yhat = self.model.predict(X)
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
        return f'NN({self.num_nodes}, l={self.l}, lr={self.lr})'

# total acceptable of moving from N to Np given data XY
def A(Np, N, X, Y, q=0.5):
    '''
    Acceptance ratio of moving from `N` to `Np` given data and 
    transition probability `q`
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
            dname='default_name'):
        self.num_nets = num_nets
        # maybe should bias to shrinking to avoid just overfitting?
        # or compute acceptance resid on validation data?
        self.trans_probs = trans_probs
        self.trans_options = trans_options
        self.dname = dname

    def setup_nets(self, l=10, lr=0.01):
        self.cyberspace = [NN(1, l=l, lr=lr) for i in range(self.num_nets)]

    def train(self, Xtr, Ytr, Xva=None, Yva=None, Xte=None, Yte=None, total_iters=10):
        if Xva is None:
            Xtr, XX, Ytr, YY = skms.train_test_split(Xtr,Ytr, test_size=0.5) # training
            if Xte is None:
                Xva, Xte, Yva, Yte = skms.train_test_split(XX,YY, test_size=0.5) # valid and test
            else:
                Xva = XX
                Yva = YY

        # initialize fit as though all get equal share of Y
        for j,N in enumerate(self.cyberspace):
            N.train(Xtr,Ytr/self.num_nets)

        # check initial fit
        Yh = np.sum([N.model.predict(Xte) for N in self.cyberspace], axis=0)
        self.Yte_init = np.copy(Yh)

        accepted = 0
        # setup residual array
        S_tr = np.array([N.model.predict(Xtr) for N in self.cyberspace])
        S_va = np.array([N.model.predict(Xva) for N in self.cyberspace])
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
                    Np = NN(N.num_nodes+1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG))
                    q = self.trans_probs[0]
                elif N.num_nodes-1 == 0:
                    continue # don't bother building empty model
                else:
                    Np = NN(N.num_nodes-1, weight_donor=N, l=N.l, lr=N.lr, r=np.random.randint(BIG))
                    q = self.trans_probs[1]
                Np.train(Xtr,Rtr)
                # determine if we should keep it
                if np.random.random() < A(Np, N, Xva, Rva, q):
                    self.cyberspace[j] = Np
                    accepted += 1
                    S_tr[j] = Np.model.predict(Xtr)
                    S_va[j] = Np.model.predict(Xva)
            # overall validation error at this MCMC iteration
            phi[i] = np.sqrt(np.mean((Rva - S_va[j])**2))
        self.phi = phi
        self.accepted = accepted
        self.Xte = Xte
        self.Yte = Yte

    def predict(self, X):
        return np.sum([N.model.predict(X) for N in self.cyberspace], axis=0)

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
        with open(outname,'wb') as f:
            pickle.dump(self.cyberspace, f)
