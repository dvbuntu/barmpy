import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from barmpy.barn import NN, A, BARN, BARN_bin
import warnings
warnings.filterwarnings("ignore")

class TestBARN(unittest.TestCase):
    USE_TF = False
    def setUp(self):
        # NB: random_state not fully respected by sklearn
        self.model = BARN(dname='test',
                num_nets=10, 
                random_state=42,
                epochs=100,
                act='logistic',
                normalize=False,
                l=1,
                ndraw=1,
                use_tf=self.USE_TF)
        self.model.setup_nets(n_features_in_=2)
        # Setup linear relationship as test data, scaled
        self.n = 1000
        self.Xo = np.arange(2*self.n).reshape((self.n,2))
        self.Yo = self.Xo[:,0] + 2*self.Xo[:,1]
        scale_x = PCA(n_components=self.Xo.shape[1], whiten=False)
        scale_x.fit(self.Xo)
        self.X = scale_x.transform(self.Xo)
        #Xtr = scale_x.transform(self.Xo)
        scale_y = StandardScaler() # no need to PCA
        scale_y.fit(self.Yo.reshape((-1,1)))
        self.Y = scale_y.transform(self.Yo.reshape((-1,1))).reshape(-1)

    def test_train_batchmeans(self):
        # Test running of train and batch means analysis
        self.model.fit(self.X, self.Y)
        var = self.model.batch_means(num_batch=2,batch_size=5, burn=0, outfile='')
        # actual value varies considerably since sklearn not deterministic
        #self.assertAlmostEqual(var, 5216.536491)

    def test_fit(self):
        # train the entire ensemble
        self.model.n_iter = 40
        self.model.fit(self.X, self.Y)
        # check reasonable fit, now fully deterministic
        pred = self.model.predict(self.X)
        np.testing.assert_allclose(pred, self.Y,
                               rtol=2, atol=0.5)

    def test_bin(self):
        # NB: random_state not fully respected by sklearn
        self.model_bin = BARN_bin(dname='test',
                num_nets=10, 
                random_state=42,
                epochs=100,
                act='logistic',
                normalize=False,
                l=1,
                use_tf=self.USE_TF)
        self.model_bin.setup_nets(n_features_in_=2)
        self.Ybin = 1*(self.Y>0)
        self.model_bin.n_iter = 40
        self.model_bin.fit(self.X, self.Ybin)
        # check reasonable fit, should be fully accurate
        pred = self.model_bin.predict(self.X)
        np.testing.assert_allclose(np.round(pred), self.Ybin,
                               rtol=0, atol=0)


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
        model = BARN(num_nets=1, normalize=False, callbacks=callbacks, n_iter=n_iter)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.n_iter, n_iter)

    def test_trans_enough(self):
        callbacks = {BARN.trans_enough:{'check_every':1,
                                        'skip_first':4,
                                        'ntrans':4}}
        model = BARN(num_nets=4, normalize=False, callbacks=callbacks)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.ntrans_iter[-1], 10)

    def test_stable_dist(self):
        callbacks = {BARN.stable_dist:{'check_every':1,
                                        'skip_first':4,
                                        'tol':.2}}
        model = BARN(num_nets=10, normalize=False, callbacks=callbacks, n_iter=100)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 3)
        self.assertLess(model.n_iter, 100)

    def test_rfwsr(self):
        callbacks = {BARN.rfwsr:{'check_every':5,
                                        'skip_first':5,
                                        't':1,
                                        'eps':1}}
        model = BARN(num_nets=10, normalize=False, callbacks=callbacks, n_iter=100)
        model.fit(self.X, self.Y)
        self.assertGreaterEqual(model.n_iter, 4)
        self.assertLess(model.n_iter, 100)

    def test_multidraw(self):
        model = BARN(num_nets=4, n_iter=100, normalize=False,
                     burn=50, ndraw=10)
        model.fit(self.X, self.Y)
        self.assertEqual(len(model.saved_draws), 10)
        pred = model.predict(self.X)
        np.testing.assert_allclose(pred, self.Y,
                               rtol=0.01, atol=0.5)

    def test_normalizer(self):
        model = BARN(num_nets=10, n_iter=400, normalize=True,
                     burn=50, ndraw=1)
        model.fit(self.Xo, self.Yo)
        pred = model.scale_y.transform(model.predict(self.Xo).reshape((-1,1))).reshape(-1)
        np.testing.assert_allclose(pred, self.Y,
                               rtol=0.01, atol=0.2)

    def test_interval(self):
        model = BARN(num_nets=4, n_iter=100, normalize=False,
                     burn=50, ndraw=10)
        model.fit(self.X, self.Y)
        lower, upper = model.predict_interval(self.X)
        pred = model.predict(self.X)
        np.testing.assert_array_less(lower, pred)
        np.testing.assert_array_less(pred, upper)

class TestBARN_TF(TestBARN):
    USE_TF = True


if __name__ == '__main__':
    unittest.main(warnings="ignore")
