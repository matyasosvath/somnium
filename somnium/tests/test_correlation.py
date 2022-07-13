#!/usr/bin/env python

import unittest
from assumption.assumption import Assumption

from correlation.correlation import Correlation
import numpy as np

from variable import Variable
from visualization.ivisualization import IVisualize
#from visualization.visualization import Visualize

class TestVariable(unittest.TestCase):

    def setUp(self):
        self.assmp = Assumption()
        self.vis = IVisualize()

    def test_correlation(self):
        # Arrange
        x = Variable([1,2,3,4,5,6,7,8,9], name="Test1")
        y = Variable([1,2,3,4,5,6,7,8,9], name="Test2")

        corr = Correlation(self.assmp, self.vis, x,y)
        # Act
        corr.correlate()
        print(corr.print_result())

        # Assert
        self.assertEqual(0, 0)


if __name__ == '__main__':
    unittest.main()
