#!/usr/bin/env python

import numpy as np 
import pandas as pd
from sklearn.utils import shuffle


class PermutationTest:
        def __init__(self, *data):
                self.data = data
                self.actual = self.test_stat(data)

        def permutation(self):
                v1, v2 = self.data[0], self.data[1]

                # for i in self.data[2:]:
                #         try:
                #                 v = i
                #         except IndexError:
                #                 pass

                pool = np.array(self.data).flatten()
                # print(f"'pool':{pool}")
                data = shuffle(pool)
                # print(f'Shuffled Data: {data}')
                
                v1 = self.resample(data, size=len(v1), replace=False)
                v2 = self.resample(data, size=len(v2), replace=False)
                return v1, v2

        def resample(self, x, size, replace = False):
                 return np.random.choice(x, size=size, replace=replace)


        def pvalue(self, iter = 1000):
                self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]
                # print(f' Observed Difference: {self.actual}')

                count = sum(1 for i in self.permute_dist if i >= self.actual)
                return count/iter




class TwoGroups(PermutationTest):

        def test_stat(self, data):
                v1, v2 = data
                return abs(v1.mean() - v2.mean())
                # return abs(v1.mean() - v2.mean())


class MultipleGroups(PermutationTest):
        pass


if __name__ == '__main__':
        np.random.seed(42)
        v1 = np.random.randint(3,10, size=30)
        v2 = np.random.randint(10, 15, size=30)

        k = TwoGroups(v1,v2)

        print(k.pvalue(), k.actual)



