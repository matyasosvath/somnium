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


class PermuteCorr(PermutationTest):
    '''Testing the significance of r that makes no distributional assumptions.'''

    def test_stat(self, data):  # Association
        x,y = data
        test_stat = covariance(x,y)/(x.std() * y.std())
        # test_stat = np.corrcoef(x,y)[0][1]
        # test_stat = pd.Series(x).corr(pd.Series(y))
        return test_stat





class MultiGroup(PermutationTest):
    """
    ANOVA is used to test differences among means by analyzing variance.

    Test uses resampling procedure (without making any distributional assumptions).

    H_0: All X population means are equal (omnibus hypothesis).
    H_1: At least one population mean is different from at least one other mean.
    """
    def test_stat(self, data):
        if len(data) <= 2:
            raise TypeError(f'In case of two groups, test with t-test (see there).')

        #data = [np.array(datum, dtype=float) for datum in data]

        # MSE
        # Mean of sample variances
        mse = np.mean([data[i].var() for i in range(len(data))])

        # MSB
        # Compute means
        # Compute variance of means
        # Multiply by n
        msb = np.var([data[i].mean() for i in range(len(data))]) * data[0].shape[0]

        # F-ratio
        f = msb/mse
        return f

        # alldata = np.concatenate(data, axis=1)

        # bign = alldata.shape[axis]

        # print(alldata, bign)
        # print('test')




#############################


class MannWhitneyU:
    """
    Rank randomization tests for differences in central tendency. Also called Wilcoxon rank-sum test
    """

    def __init__(self):
        pass

    def test_stat(self, *data, pvalue=True, alternative='two-sided', distribution='different'):
        x, y = data

        if not isinstance(x, pd.core.series.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.core.series.Series):
            y = pd.Series(y)

        n1 = len(x)
        n2 = len(y)
        data = pd.concat((x, y)).rank()
        x, y = data[:n1], data[n1:]

        if distribution == 'different':

            all_rank_sum = n1*(n1+1)/2  # all rank sum
            R_1 = sum(x)  # sum rank of group 1
            R_2 = sum(y)  # or all_rank_sum - R_x

            U1 = R_1 - ((n1*(n1+1))/2)

            U2 = R_2 - ((n2*(n2+1))/2)
            U = min(U1, U2)

            Umean = (n1*n2)/2
            Ustd = np.sqrt(n1*n2*(n1+n2+1)/12)

        elif distribution == 'identical':  # identical distribution
            x, y = data
            x_median = pd.Series(x).median()
            y_median = pd.Series(y).median()
            # return abs(x_median - y_median) # difference between sum of ranks; two-tailed
        else:
            raise Exception(
                'You should specify the distribution parameter. Available parameters: "identical", "different".')

        def pvalue(alternative=alternative):
            # For ldatae samples, U is approximately normally distributed. In that case, the standardized value equals to
            z = (U - Umean) / Ustd
            if alternative == 'two-sided':
                p = 2*stats.norm.sf(abs(z))
            elif alternative == 'one-sided':
                p = 1 - stats.norm.cdf(abs(z))
            else:
                raise Exception('Hypothesis test should be one or two sided.')
            return p
        if pvalue:
            p = pvalue()
        return tuple((U, p))



if __name__ == '__main__':
    x = np.random.randint(10, 15, size=30)
    y = np.random.randint(10, 15, size=30)
    z = np.random.randint(10, 15, size=30)
    w = np.random.randint(13, 18, size=30)

    anova = MultiGroup(x, y,z)#,w)

    print(anova.pvalue(), anova.actual)

    twogroup = DiffTwoMeans(x,z)
    print(twogroup.pvalue(), twogroup.actual)

    corr = PermuteCorr(x,w)
    print(corr.actual, corr.pvalue())
    from scipy.stats import pearsonr
    print(pearsonr(x, y))


    # print('Mann Whitney')
    # U, p = MannWhitneyU().test_stat(x, y,  alternative='two-sided')
    # print(U, p)

    # print('\n')

    # print('Scipy Stats Mann Whitney')
    # stat, pvalue = stats.mannwhitneyu(x, y, alternative='two-sided')
    # print(stat, pvalue)
    # stat, pvalue = stats.mannwhitneyu(x, y, alternative='less')
    # print(stat, pvalue)
