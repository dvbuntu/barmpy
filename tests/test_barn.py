import unittest
import numpy as np

from barmpy.barn import NN, A, BARN

class TestBARN(unittest.TestCase):
    def setUp(self):
        self.model = BARN(num_nets=3, dname='test')
        self.model.setup_nets()
        self.n = 100
        self.X = np.arange(2*self.n).reshape((self.n,2))
        self.Y = self.X[:,0] + 2*self.X[:,1]

    def test_train_batchmeans(self):
        self.model.train(self.X, self.Y)
        self.model.batch_means(num_batch=2,batch_size=5, burn=0)

if __name__ == '__main__':
    unittest.main()
