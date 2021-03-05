#!/usr/bin/env python

from scipy import stats
import pandas as pd
import  numpy as np
from sklearn.utils import shuffle


# My modules
from hypothesis import PermutationTest

class PermuteCorr(PermutationTest):
    '''Testing the significance of r that makes no distributional assumptions.'''
    def test_stat(self, data): # Association
        test_stat = self.pearson()
        return test_stat


class MannWhitneyU: 
    """
    Rank randomization tests for differences in central tendency. Also called Wilcoxon rank-sum test
    """

    def __init__(self):
        pass

    def test_stat(self, *data, pvalue=True, alternative='two-sided', distribution='different'):
        x,y = data
        
        if not isinstance(x, pd.core.series.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.core.series.Series):
            y = pd.Series(y)

        n1 = len(x)
        n2 = len(y)
        data = pd.concat((x,y)).rank()
        x,y = data[:n1], data[n1:]

        if distribution == 'different':

            all_rank_sum = n1*(n1+1)/2 # all rank sum
            R_1 = sum(x) # sum rank of group 1
            R_2 = sum(y) # or all_rank_sum - R_x

            U1 = R_1 - ((n1*(n1+1))/2)

            U2 = R_2 - ((n2*(n2+1))/2)
            U = min(U1,U2)

            Umean = (n1*n2)/2
            Ustd = np.sqrt(n1*n2*(n1+n2+1)/12)

        elif distribution == 'identical': # identical distribution
            x,y = data
            x_median = pd.Series(x).median()
            y_median = pd.Series(y).median() 
            # return abs(x_median - y_median) # difference between sum of ranks; two-tailed
        else:
            raise Exception('You should specify the distribution parameter. Available parameters: "identical", "different".')
        
        def pvalue(alternative = alternative):
            z = (U - Umean)/ Ustd # For large samples, U is approximately normally distributed. In that case, the standardized value equals to
            if alternative == 'two-sided':
                p = 2*stats.norm.sf(abs(z))
            elif alternative == 'one-sided':
                p = 1 - stats.norm.cdf(abs(z))
            else:
                raise Exception('Hypothesis test should be one or two sided.')
            return p
        if pvalue:
            p = pvalue()
        return tuple((U,p))




if __name__== '__main__':
    import numpy as np
    np.random.seed(42)
    x  = np.random.randint(10,15, size=30)
    y = np.random.randint(11, 15, size=30)

    print('Mann Whitney')
    U, p = MannWhitneyU().test_stat(x,y,  alternative='two-sided')
    print(U, p)
    
    print('\n')

    print('Scipy Stats Mann Whitney')
    stat, pvalue = stats.mannwhitneyu(x, y, alternative='two-sided')
    print(stat,pvalue)
    stat, pvalue = stats.mannwhitneyu(x, y, alternative='less')
    print(stat,pvalue)
