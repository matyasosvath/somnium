#!/usr/bin/env python

import unittest

from assumption.abstract_assumption import AbstractAssumption
import numpy as np

class TestVariable(unittest.TestCase):

    def setUp(self):
        self.assmp = AbstractAssumption()

    def test_normality_test(self):
        # Arrange
        mu, sigma = 0, 0.1 # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        #print(s)

        expected_p_value = 0.05
        expected_is_normal = True
        # Act
        actual = self.assmp.normality_test(s, name="Test")
        print(actual)
        # Assert
        self.assertEqual(actual["is_normal"], expected_is_normal)
        self.assertGreaterEqual(actual["p_value"], expected_p_value)

    def test_assumption_property(self):
        # Arrange
        expected = dict()

        # Act
        actual = self.assmp.assumptions

        # Assert
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
