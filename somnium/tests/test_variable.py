#!/usr/bin/env python

import unittest

from variable import Variable

class TestVariable(unittest.TestCase):

    def setUp(self):
        self.var1 = Variable([0,1,2,3,4,5,6,7,8,9], name="Test1")
        self.var2 = Variable([1,1,1,2,2,2], name="Test2")
        self.var3 = Variable([1,2,3,4,5,6,5,5,4,6])                

    def test_variable_len(self):
        # Arrange
        expected1 = 10
        expected2 = 6
        expected3 = 10

        # Act
        actual1 = self.var1.N
        actual2 = self.var2.N
        actual3 = self.var3.N

        # Assert
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)

    def test_variable_name(self):
        # Arrange
        expected1 = "Test1"
        expected2 = "Test2"
        expected3 = None

        # Act
        actual1 = self.var1.name
        actual2 = self.var2.name
        actual3 = self.var3.name

        # Assert
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)


    def test_variable_data_type(self):
        # Arrange
        expected1 = "CONTINUOUS"
        expected2 = "NOMINAL"
        expected3 = "ORDINAL"

        # Act
        actual1 = self.var1.type
        actual2 = self.var2.type
        actual3 = self.var3.type

        # Assert
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)

    def test_variable_values(self):
        # Arrange
        expected1 = [0,1,2,3,4,5,6,7,8,9]
        expected2 = [1,1,1,2,2,2]
        expected3 = [1,2,3,4,5,6,5,5,4,6]

        # Act
        actual1 = self.var1.values
        actual2 = self.var2.values
        actual3 = self.var3.values

        # Assert
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)

    def test_variable_error_on_new_attribute_assignemtn(self):
        self.assertRaises(AttributeError, lambda: self.var1.some_type)

    def test_variable_index(self):
        # Arrange
        expected1 = 0
        expected2 = 2
        expected3 = 5

        # Act
        actual1 = self.var1[0]
        actual2 = self.var2[5]
        actual3 = self.var3.values[4]

        # Assert
        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)
        self.assertEqual(actual3, expected3)


if __name__ == '__main__':
    unittest.main()
