import unittest
import numpy as np

from barmpy.barn import NN, A, BARN
import warnings
warnings.filterwarnings("ignore")


class TestNN(unittest.TestCase):
    def setUp(self):
        # Create ensemble of 3 setworks with 1,2,3 neurons
        self.cyberspace = [NN(num_nodes=i+1, r=i) for i in range(3)]
        # Setup linear relationship as test data
        self.X = np.arange(20).reshape((10,2))
        self.Y = self.X[:,0] + 2*self.X[:,1]
        self.sigma = np.std(self.Y)

    def test_logprior(self):
        priors = [nn.log_prior() for nn in self.cyberspace]
        # Numerical priors given defaults and small test data
        self.assertAlmostEqual(priors,
                [-7.697414907005954,
                 -6.087976994571854,
                 -4.884004190245918])

    def test_train_donate(self):
        # Train just the first network
        self.cyberspace[0].train(self.X,self.Y)
        # Overwrite NN1 with NN0 weights where applicable
        donor = self.cyberspace[0]
        donee = self.cyberspace[1]
        donee.accept_donation(donor.num_nodes,
                              donor.model.coefs_,
                              donor.model.intercepts_)
        # Check each chunk of weights, layer 1, intercepts, output, intercepts
        np.testing.assert_allclose(donee.model.coefs_[0][:,0].reshape(-1),
                               donor.model.coefs_[0].reshape(-1),
                               rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(donee.model.coefs_[1][0].reshape(-1),
                               donor.model.coefs_[1].reshape(-1),
                               rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(donee.model.intercepts_[0][0].reshape(-1),
                               donor.model.intercepts_[0].reshape(-1),
                               rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(donee.model.intercepts_[1].reshape(-1),
                               donor.model.intercepts_[1].reshape(-1),
                               rtol=1e-4, atol=1e-4)

    def test_train(self):
        # Ensure training process runs, but don't check weights
        self.cyberspace[1].train(self.X,self.Y)

    def test_ll(self):
        # Train the last network
        self.cyberspace[2].train(self.X,self.Y)
        # Compute LL and compare to known value for data/weights
        ll = self.cyberspace[2].log_likelihood(self.X, self.Y, self.sigma)
        self.assertAlmostEqual(ll, -42.65804602606022)

    def test_bad_nodes(self):
        # use string instead of number, fails on training
        try:
            ## NB: sklearn happy to do this
            nn_str = NN(num_nodes='4')
            nn_str.train(self.X, self.Y)
        except TypeError:
            pass
        # float instead of int (and not just float dtype)
        try:
            ## NB: sklearn happy to do this
            nn_flt = NN(num_nodes=4.5)
            nn_flt.train(self.X, self.Y)
        except TypeError:
            pass
        # None as input
        try:
            ## NB: sklearn happy to do this
            nn_none = NN(num_nodes=None)
            nn_none.train(self.X, self.Y)
        except TypeError:
            pass
            
    def test_binary(self):
        nn = NN(binary=True)
        Ybin = np.round(np.arange(10)/10)
        nn.train(self.X, Ybin)
        pred = nn.predict(self.X)
        # verify range
        self.assertLessEqual(np.max(pred), 1)
        self.assertLessEqual(0, np.min(pred))


if __name__ == '__main__':
    unittest.main()
