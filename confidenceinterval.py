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

    def pvalue(self, iter=1000, ci=True, ci_level=95):

        # Calculate permuted distribution
        self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]

        # P-value
        count = sum(1 for i in self.permute_dist if i >= self.actual)
        print(count)
        # Confidence Interval
        statistics = sorted(self.permute_dist)
        #print(statistics)
        # Trim endpoints of resampled CI
        trim = ((1 - (ci_level/100))/2)
        endpoints = int(trim*1000)
        trimmed_ci = statistics[endpoints:-endpoints]
        lower, upper = min(trimmed_ci), max(trimmed_ci)



        return np.round(count/iter, 3), lower, upper


class DiffTwoMeans(PermutationTest):
    """
    Significance test for the difference between two groups.
    Makes no distributional assertion.
    """
    def test_stat(self, data):
        if len(data) > 2:
            raise TypeError(f'In case of more than two groups, test with ANOVA (see there).')

        v1, v2 = data
        return abs(v1.mean() - v2.mean())
        # return abs(v1.mean() - v2.mean())



if __name__ == '__main__':
    x = np.random.randint(10, 15, size=30)
    y = np.random.randint(10, 15, size=30)
    z = np.random.randint(13, 15, size=30)
    w = np.random.randint(16, 25, size=30)

    d = DiffTwoMeans(x,y)
    print(d.pvalue())
    print(d.actual)
    print(x.mean()- y.mean())





## -------------


# class ConfidenceInterval:
# 	def __init__(self):
# 		pass

# 	def bootstrapped(self, data):
# 		pass



# def interval(*variables, ci=95, method='bootstrap', iters=1000):
#     x,y = variables
#     means = sorted(list((np.mean(np.random.choice(x, size=len(x), replace=True)) for i in range(iters))))
#     # for i in range(iters):
#     #     x = np.random.choice(x, size=len(x), replace=True)
#     #     means.append(x.mean())
    

#     # Trim endpoints of resampled CI
#     trim = ((1 - (ci/100))/2)
#     endpoints = int(trim*1000)
#     trimmed_ci = means[endpoints:-endpoints]
#     lower, upper = min(trimmed_ci), max(trimmed_ci)

#     print(trim, endpoints, trimmed_ci, lower, upper)

#     # or 
#     trimmed_ci = means[25:]

#     return lower, upper
