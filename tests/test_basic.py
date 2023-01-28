import unittest
import numpy as np

from barmpy.submodule import function 

class TestBasic(unittest.TestCase):

    def setUp(self):
        self.data = np.arange(10)

    def test_function(self):
        result = function(self.data)
        self.assertEqual(result, np.arange(10)+1)

if __name__ == '__main__':
    unittest.main()
