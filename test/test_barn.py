import unittest
import numpy as np

from barmpy.barn import NN, A, BARN
import warnings
warnings.filterwarnings("ignore")

class TestBARN(unittest.TestCase):
    USE_TF = False
    def setUp(self):
        # NB: random_state not fully respected by sklearn
        self.model = BARN(num_nets=10, dname='test',
                random_state=42,
                epochs=1,
                use_tf=self.USE_TF)
        self.model.setup_nets(n_features_in_=2)
        # Setup linear relationship as test data
        self.n = 1000
        self.X = np.arange(2*self.n).reshape((self.n,2))
        self.Y = self.X[:,0] + 2*self.X[:,1]

    def test_train_batchmeans(self):
        # Test running of train and batch means analysis
        self.model.fit(self.X, self.Y)
        var = self.model.batch_means(num_batch=2,batch_size=5, burn=0, outfile='')
        # actual value varies considerably since sklearn not deterministic
        #self.assertAlmostEqual(var, 5216.536491)

    def test_fit(self):
        # train the entire ensemble
        self.model.n_iter = 20
        self.model.fit(self.X, self.Y)
        # check reasonable fit, now fully deterministic
        pred = self.model.predict(self.X)
        np.testing.assert_allclose(self.Y, pred,
                               rtol=2, atol=10)

    def test_trans(self):
        # need list of numbers, not strictly summing to 1
        # b/c they are all relative
        ## str
        try:
            model = BARN(num_nets=10, trans_probs='ab')
        except TypeError:
            pass
        # check for same number of trans probs and trans options 
        try:
            model = BARN(num_nets=10,
                    trans_probs=[0.33],
                    trans_options=['grow','shrink'],
                    )
        except IndexError:
            pass

    def test_bad_nnet(self):
        # should be int, but in case it's not, throw exception
        ## str
        try:
            model = BARN(num_nets='10')
            model.setup_nets()
        except TypeError:
            pass
        ## float
        try:
            model = BARN(num_nets=10.5)
            model.setup_nets()
        except TypeError:
            pass

    def test_improvement(self):
        callbacks = {BARN.improvement:{'check_every':1,'skip_first':4}}
        n_iter=100
        model = BARN(num_nets=1, callbacks=callbacks, n_iter=n_iter)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.n_iter, n_iter)

    def test_trans_enough(self):
        callbacks = {BARN.trans_enough:{'check_every':1,
                                        'skip_first':4,
                                        'ntrans':4}}
        model = BARN(num_nets=4, callbacks=callbacks)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.ntrans_iter[-1], 4)

    def test_stable_dist(self):
        callbacks = {BARN.stable_dist:{'check_every':1,
                                        'skip_first':4,
                                        'tol':.2}}
        model = BARN(num_nets=10, callbacks=callbacks, n_iter=100)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.n_iter, 100)

    def test_rfwsr(self):
        callbacks = {BARN.rfwsr:{'check_every':5,
                                        'skip_first':5,
                                        't':1,
                                        'eps':1}}
        model = BARN(num_nets=10, callbacks=callbacks, n_iter=100)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 4)
        self.assertLess(model.n_iter, 100)

class TestBARN_TF(TestBARN):
    USE_TF = True


if __name__ == '__main__':
    unittest.main(warnings="ignore")
