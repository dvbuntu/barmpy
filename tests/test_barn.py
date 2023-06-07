import unittest
import numpy as np

from barmpy.barn import NN, A, BARN

class TestBARN(unittest.TestCase):
    USE_TF = False
    def setUp(self):
        # NB: random_state not fully respected by sklearn
        self.model = BARN(num_nets=10, dname='test',
                random_state=42,
                x_in=2, 
                use_tf=self.USE_TF)
        self.model.setup_nets(epochs=1)
        # Setup linear relationship as test data
        self.n = 1000
        self.X = np.arange(2*self.n).reshape((self.n,2))
        self.Y = self.X[:,0] + 2*self.X[:,1]

    def test_train_batchmeans(self):
        # Test running of train and batch means analysis
        self.model.train(self.X, self.Y)
        var = self.model.batch_means(num_batch=2,batch_size=5, burn=0, outfile='')
        # actual value varies considerably since sklearn not deterministic
        #self.assertAlmostEqual(var, 5216.536491)

    def test_fit(self):
        # train the entire ensemble
        self.model.train(self.X, self.Y, total_iters=20)
        # check reasonable fit, not fully deterministic
        pred = self.model.predict(self.X)
        np.testing.assert_allclose(self.Y, pred,
                               rtol=2, atol=10)

    def test_trans(self):
        # need list of numbers, not strictly summing to 1
        # b/c they are all relative
        ## str
        try:
            model = BARN(num_nets=10, trans_probs='ab', x_in=2)
        except TypeError:
            pass
        # check for same number of trans probs and trans options 
        try:
            model = BARN(num_nets=10,
                    trans_probs=[0.33],
                    trans_options=['grow','shrink'],
                    x_in=2)
        except IndexError:
            pass

    def test_bad_nnet(self):
        # should be int, but in case it's not, throw exception
        ## str
        try:
            model = BARN(num_nets='10', x_in=2)
            model.setup_nets(epochs=1)
        except TypeError:
            pass
        ## float
        try:
            model = BARN(num_nets=10.5, x_in=2)
            model.setup_nets(epochs=1)
        except TypeError:
            pass

class TestBARN_TF(TestBARN):
    USE_TF = True


if __name__ == '__main__':
    unittest.main()
