#!/usr/bin/env python

from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle



from descriptive import covariance


class PermutationTest:

    def __init__(self, *data):
        self.data = data
        self.actual = self.test_stat(data)

    def permutation(self):

        
        data = [np.array(datum, dtype=float) for datum in self.data]
        elemszamok = [data[i].shape[0] for i in range(len(data))] # returns list
        csoportszam = len(elemszamok)


        pool = np.concatenate(data, axis=0)
        #print(f"'pool':{pool}")
        pool = shuffle(pool) # shuffle pool in-place
        #print(f'Shuffled Data: {data}')


        # need list of arrays
        data = [self.resample(pool, size=elemszamok[i]) for i in range(csoportszam)]
        return data

    def resample(self, x, size, replace=False):
        return np.random.choice(x, size=size, replace=replace)

    def pvalue(self, iter=1000):
        self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]
        #print(self.permute_dist)
        count = sum(1 for i in self.permute_dist if i >= self.actual)
        return np.round(count/iter, 3)




class ConfidenceInterval:
	def __init__(self):
		pass

	def bootstrapped(self, data):
		pass



def interval(*variables, ci=95, method='bootstrap', iters=1000):
    x,y = variables
    means = sorted(list((np.mean(np.random.choice(x, size=len(x), replace=True)) for i in range(iters))))
    # for i in range(iters):
    #     x = np.random.choice(x, size=len(x), replace=True)
    #     means.append(x.mean())
    

    # Trim endpoints of resampled CI
    trim = ((1 - (ci/100))/2)
    endpoints = int(trim*1000)
    trimmed_ci = means[endpoints:-endpoints]
    lower, upper = min(trimmed_ci), max(trimmed_ci)

    print(trim, endpoints, trimmed_ci, lower, upper)

    # or 
    trimmed_ci = means[25:]

    return lower, upper
