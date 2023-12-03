import unittest
import numpy as np

DO_TF_TEST = True
try:
    from barmpy.barn import TF_NN as NN
except ImportError:
    DO_TF_TEST = False
import warnings
warnings.filterwarnings("ignore")

if DO_TF_TEST:
    class TestNN(unittest.TestCase):
        def setUp(self):
            # Create ensemble of 3 setworks with 1,2,3 neurons
            self.cyberspace = [NN(num_nodes=i+1, r=i, epochs=1, x_in=2) for i in range(3)]
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
            donor_coefs_, donor_intercepts_ = donor.get_weights()
            donee.accept_donation(donor.num_nodes,
                                  donor_coefs_,
                                  donor_intercepts_)
            donee_coefs_, donee_intercepts_ = donee.get_weights()
            # Check each chunk of weights, layer 1, intercepts, output, intercepts
            np.testing.assert_allclose(donee_coefs_[0][:,0].reshape(-1),
                                   donor_coefs_[0].reshape(-1),
                                   rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(donee_coefs_[1][0].reshape(-1),
                                   donor_coefs_[1].reshape(-1),
                                   rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(donee_intercepts_[0][0].reshape(-1),
                                   donor_intercepts_[0].reshape(-1),
                                   rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(donee_intercepts_[1].reshape(-1),
                                   donor_intercepts_[1].reshape(-1),
                                   rtol=1e-4, atol=1e-4)

        def test_train(self):
            # Ensure training process runs, but don't check weights
            self.cyberspace[1].train(self.X,self.Y)

        def test_ll(self):
            # Train the last network
            self.cyberspace[2].train(self.X,self.Y)
            # Compute LL and compare to known value for data/weights
            ll = self.cyberspace[2].log_likelihood(self.X, self.Y, self.sigma)
            self.assertAlmostEqual(ll, -53.08551, places=1)

        def test_bad_nodes(self):
            # use string instead of number, fails on training
            try:
                ## NB: sklearn happy to do this
                nn_str = NN(num_nodes='4', x_in=2, epochs=1)
                nn_str.train(self.X, self.Y)
            except TypeError:
                pass
            # float instead of int (and not just float dtype)
            try:
                ## NB: sklearn happy to do this
                nn_flt = NN(num_nodes=4.5, x_in=2, epochs=1)
                nn_flt.train(self.X, self.Y)
            except TypeError:
                pass
            # None as input
            try:
                ## NB: sklearn happy to do this
                nn_none = NN(num_nodes=None, x_in=2, epochs=1)
                nn_none.train(self.X, self.Y)
            except TypeError:
                pass
                


if __name__ == '__main__':
    unittest.main()
