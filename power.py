#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import unittest
from sklearn.utils import shuffle


if len(sys.argv) > 1:
    if sys.argv[1] == 'statistical_rigor':
        power = sys.argv[1]
        alpha = sys.argv[2]
        effect_size = sys.argv[3]
        sample_size = sys.argv[4]
    else:
        power = sys.argv[1]  # 0.8
        alpha = sys.argv[2]  # 0.05
        mean_x = sys.argv[3]
        mean_y = sys.argv[4]
        sample_size = sys.argv[5]


class PermutationTest:

    def __init__(self, *data):
        self.data = data
        self.actual = self.test_stat(data)

    def permutation(self):
        data = [np.array(datum, dtype=float) for datum in self.data]
        elemszamok = [data[i].shape[0]
                      for i in range(len(data))]  # returns list
        csoportszam = len(elemszamok)

        pool = np.concatenate(data, axis=0)
        # print(f"'pool':{pool}")
        pool = shuffle(pool)  # shuffle pool in-place
        #print(f'Shuffled Data: {data}')

        # need list of arrays
        data = [self.resample(pool, size=elemszamok[i])
                for i in range(csoportszam)]
        return data

    def resample(self, x, size, replace=False):
        return np.random.choice(x, size=size, replace=replace)

    def pvalue(self, iter=1000):
        self.permute_dist = [self.test_stat(
            self.permutation()) for x in range(iter)]
        #print(self.permute_dist)
        count = sum(1 for i in self.permute_dist if i >= self.actual)
        return np.round(count/iter, 3)


class Power(PermutationTest):
    """
    Documentation
    """

    def test_stat(self, data, method="mean_difference"):
        if method == 'mean_difference':
            if len(data) > 2:
                raise TypeError("More than two groups")

            x, y = data
            return abs(x.mean() - y.mean())

        if method == 'cohen':
            pass

        if method == 'need more tests':
            pass

    def estimate_power(self, num_runs=101):
        x, y = self.data
        power_count = 0

        for i in range(num_runs):
            resample_x = np.random.choice(x, len(x), replace=True)
            resample_y = np.random.choice(y, len(y), replace=True)

            p = self.pvalue()
            if p < 0.05:
                power_count += 1
        print('hello')
        return power_count/num_runs


if __name__ == '__main__':
    x = np.random.randint(10, 15, size=30)
    y = np.random.randint(16, 30, size=30)

    p = Power(x,y)
    print(p.pvalue())
    print(p.estimate_power())
    
