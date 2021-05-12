#!/usr/bin/env python

from scipy import stats
import scipy
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle


#from descriptive import covariance


class PermutationTest:

    def __init__(self, *data):
        self.data = data        
        self.actual = self.test_stat(data)

    def permutation(self):
        data = [np.array(datum, dtype=float) for datum in self.data]
        elemszamok = [data[i].shape[0] for i in range(len(data))] # returns list
        csoportszam = len(elemszamok)

        pool = np.concatenate(data, axis=0)
        pool = shuffle(pool) # shuffle pool in-place

        # need list of arrays
        data = [self.resample(pool, size=elemszamok[i]) for i in range(csoportszam)]
        return data

    def resample(self, x, size, replace=False):
        return np.random.choice(x, size=size, replace=replace)

    def pvalue(self, iter=1000, ci=True, ci_level=95):
        # Permuted sampling distribution
        self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]

        # P-value
        count = sum(1 for i in self.permute_dist if i >= self.actual)
        print(count)

        #TODO compute confidence interval using your own percentile function
        # Bootstraped [bs] Confidence Interval
        if ci:
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


class ConfidenceInterval:
    def __init__(self, *data, alpha=0.05, ci_level=.95):
        self.data = data
        self.alpha = alpha
        self.ci_level = ci_level


    def forMean(self, variance='unknown', **kwargs):
        """
        Confidence Interval for the Mean
        Defaults to unknown variance, use t-distribution
        """
        x = np.array(self.data)
        print(x.size)
        print(x)
        if variance == 'unknown':
            sem = x.std()/ np.sqrt(x.size)
            df = x.size - 1
            t_cl = stats.t.interval(self.alpha,df)[1]
            print(sem,df,t_cl)
            lower_limit = x.mean() - t_cl * sem
            upper_limit = x.mean() + t_cl * sem
            #print(np.round(lower_limit,3), np.round(upper_limit,3))
            return (np.round(lower_limit,3), np.round(upper_limit,3))


        if variance == 'known':
            var = kwargs['var']
            sem = var / np.sqrt(x.shape[0])
            z_cl = scipy.stats.norm.ppf(ci_level)
            lower_limit = x.mean() - z_cl * sem
            upper_limit = x.mean() + z_cl * sem
            #print(lower, upper)
            return (upper_limit, lower_limit)


            
    def forDifferenceBetweenmeans(self, equal_n = True):
        x,y = self.data
        x_n, y_n = x.shape[0], y.shape[0]
        print(x,y)
        if equal_n:
            # Standard error (equal sample size)
            mean_diff = x.mean(), y.mean()
            mse = (x.var()**2 + y.var()**2)/2

            SE = np.sqrt(2*mse/x_n)
            t_cl = scipy.stats.t.interval(alpha,df)
            df = x_n - 1 + y_n - 1


            # Compute CI
            lower = mean_diff - t_cl * SE
            upper = mean_diff + t_cl * SE
            return (lower, upper)
        else:

            SSE = sum((i - x.mean())**2 for i in x) + sum((i - y.mean())**2 for i in y)
            df = x_n - 1 + y_n - 1

            MSE = SSE/df

            n_h = 2/ (1/x_n + 1/y_n) # harmonic mean

            mean_diff = x.mean(), y.mean()

            SE = np.sqrt(2*mse/n_h)

            t_cl = scipy.stats.t.interval(alpha,df)

            # Compute CI
            lower = mean_diff - t_cl * SE
            upper = mean_diff + t_cl * SE
            #print(np.round(lower,3), np.round(upper,3))
            return (np.round(lower,3), np.round(upper,3))


        def forCorr(self):
            """
            1. Convert r to z'
            2. Compute a confidence interval in terms of z'
            3. Convert the confidence interval back to r.
            """

            x,y = self.data

            r = np.corrcoef(x,y)

            # Step 1
            r_z = np.arctanh(r)
            r_z = 0.5 * np.log(1+r/1-r)

            # Step 2
            SE = 1/np.sqrt(x.shape[0]-3)

            z_ci = stats.norm.ppf(1-alpha/2)

            # Compute CI
            lower = r_z - z_ci * SE
            upper = r_z + z_ci * SE

            # Step 3
            lower, upper = np.tanh((lower, upper))
            return (lower, upper)



        def forProp(self,favorable_outcome):
            p = pd.value_counts(self.data, normalize=True)[favorable_outcome]
            n = self.data.shape[0]
            SE_p = np.sqrt(p*(1-p)/n)

            z_ci = stats.norm.ppf(1-alpha/2)

            # Compute CI (+ correction for estimating discrete distribution)
            lower = p - z_ci * SE_p - (0.5/n) 
            upper = p + z_ci * SE_p + (0.5/n)
            return (lower, upper)



import unittest


class TestSum(unittest.TestCase):

    z_prop = np.random.randint(0, 2, size=30)
    print(f'Prop values {z_prop}')


    #ci_class = ConfidenceInterval(x,y)

    def test_formean(self):
        np.random.seed(42)
        x = np.random.randint(5, 15, size=30)
        y = np.random.randint(10, 15, size=30)
        self.assertEqual(ConfidenceInterval(x).forMean(), (9.703, 9.763), "Should be (9.703, 9.763)")

    # def test_sum_tuple(self):
    #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()


# if __name__ == '__main__':


#     d = DiffTwoMeans(x,y)
#     print(d.pvalue())
#     print(d.actual)
#     print(x.mean()- y.mean())



