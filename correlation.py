#!/usr/bin/env python

from scipy import stats
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# My modules
from hypothesis import PermutationTest


 
class Correlation:
    """
    Computing the correlation of two (X,Y) variables. Hypothesis testing for correlation.
    Available methods: Covariance, Pearson correlation, p-value and degrees of freedom.
    """

    def __init__(self, measurement_level='continuous',
                    linearity=True, 
                    normal_dist=True, 
                    outliers=False):
        self.measurement_level = measurement_level
        self.linearity = linearity
        self.normal_dist = normal_dist
        self.outliers = outliers

    def assumptions(self, x,y):
        """Assumptions"""
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        assert x.shape[0] == y.shape[0]  # Related pairs
        if self.measurement_levelurement != 'continuous':         # Continuous variables
            raise Exception("Both level of measurement must be continuous.") # raise Exception("Message")


    def covariance(self, x,y):
        # Deviations from mean
        d_x = x - x.mean()
        d_y = y - y.mean()

        cov = np.dot(d_x,d_y)/x.shape[0] 
        return cov

    def pearson(self, x,y, significance=True):
        """Compute the Pearson correlation coefficient for two variables"""
        r = self.covariance(x,y)/(x.std() * y.std())

        # Compute p-value for correlation.
        def pvalue(x,y):
            t = r*np.sqrt(len(x) + len(y) - 2)/np.sqrt(1 - r**2) # t-statistic
            dof = len(x) - 1 + len(y) - 1 # degrees of freedom
            pvalue = stats.t.sf(t, dof)*2
            return pvalue, dof

        if significance:
            p, dof = pvalue(x,y)
            return r, p, dof
        else:
            return r
        

    def spearman(self, x,y, significance=True):
        x_rank = pd.Series(x).rank()
        y_rank = pd.Series(y).rank()

        rho = self.pearson(x_rank, y_rank, significance=False)

        def pvalue(iters=1000):
            resample = []
            for iter in range(iters):
                #x_rank_shuffled = shuffle(x_rank)
                y_rank_shuffled = shuffle(y_rank)
                resample.append(self.pearson(x_rank, y_rank_shuffled, significance=False))

            count = sum(1 for i in resample if i >= rho)
            return count/iters

        if significance:
            p = pvalue()
            return rho, p
        else:
            return rho


    
if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.randint(10,15, size=30)
    y = np.random.randint(10,15, size=30)

    c =  Correlation()

    r, pvalue, dof  = c.pearson(x,y)
    print(r, pvalue, dof)
    s_r, s_p = stats.pearsonr(x,y)
    print(f'Scipy Pearson r {s_r} and p {s_p}')
    
    
    print('\n')
    
    
    r, pvalue  = c.spearman(x,y)
    print(f'rho: {r}, p érték: {pvalue}')
    precise_rho = pd.Series(x).corr(pd.Series(y), method='spearman')

    print(f'igazi rho{precise_rho}')


