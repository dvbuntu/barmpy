import unittest
import numpy as np

from barmpy.barn import NN, A, BARN

class TestNN(unittest.TestCase):
    def setUp(self):
        self.cyberspace = [NN(num_nodes=i+1, r=i) for i in range(3)]
        self.X = np.arange(20).reshape((10,2))
        self.Y = self.X[:,0] + 2*self.X[:,1]

    def test_logprior(self):
        priors = [nn.log_prior() for nn in self.cyberspace]
        self.assertAlmostEqual(priors,
                [-7.697414907005954,
                 -6.087976994571854,
                 -4.884004190245918])

    def test_train_donate_ll(self):
        self.cyberspace[0].train(self.X,self.Y, epochs=2)
        donor = self.cyberspace[0]
        donee = self.cyberspace[1]
        donee.accept_donation(donor.num_nodes,
                              donor.model.coefs_,
                              donor.model.intercepts_)
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
        ll = self.cyberspace[0].log_likelihood(self.X, self.Y)
        self.assertAlmostEqual(ll, -35.913113112395365)


if __name__ == '__main__':
    unittest.main()
