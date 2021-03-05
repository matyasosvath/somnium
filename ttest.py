#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns 
from scipy import stats
from scipy.stats import t

import standard_error

from writer import Writer


class t_test:
    def __init__(self, variance_equal=True, normality=True, sample_size_equal = True, independent_sample=True,):
        # Checking for assumptions across all t-tests
        self.variance_equal = variance_equal # homogenity of variance
        # MISSING normal distribution
        self.independent_sample = independent_sample # each value is sampled indenpendently
        self.sample_size_equal = sample_size_equal # equal size

        self.sem = standard_error.Sem()
        self.writer = Writer()



    def t_statistic(self, statistic, sem,hypothesized_value=0):
        """Compute t-statistic"""
        return statistic - hypothesized_value/sem
        


    def one_sample(self, n1, n2, hypothesized_value=0):
        statistic = np.array(n1) - np.array(n2)
        # sem = statistic.var()/np.sqrt(len(n1)) # standard error of the statistic
        sem = self.sem.sem_mean(statistic, n1)

        t_statistic = self.t_statistic(statistic.mean(),sem)
        df = n1.shape[0] - 1 # degrees of freedom
        
        p_value = stats.t.sf(t_statistic, df) # two sided p-value

        h0_false = f"If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}. \
        The null hypothesis that the population mean difference score is zero can be rejected. \
        The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition."

        h0_not_rejected = f"The null hypothesis - that the population mean difference score is zero - cannot be rejected."

        if p_value < 0.05:
            self.writer.generic_writer(h0_false)
            print(f'If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}.')
            print('The null hypothesis that the population mean difference score is zero can be rejected.')
            print('The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition.') 
        else:
            self.writer.generic_writer(h0_not_rejected)
            print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')


    def indenpendent_sample(self, n1, n2, hypothesized_value=0):
        """Test for difference between means from two separate groups of subjects."""

        # Compute the statistic
        statistic = n1.mean() - n2.mean()

        # Compute the estimate of the standard error of the statistic
        mse = (n1.var() - n2.var())/2
        sem = np.sqrt(2*mse/n1.shape[0]) # n is equal the number of scores in each group
        # sem = S_{m_1-m_2}

        # sem = self.sem.sem_mean2(n1,n2)

        # Compute t
        t_statistic = statistic/sem
        df = (n1.shape[0]) - 1 + (n2.shape[0] - 1)
        # Compare areas of the t distribution
        p_value = stats.t.sf(t_statistic, df)

        # Interpretation
        if p_value < 0.05:
            print(f'If the treatment/experiment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low (t={t_statistic} p={p_value}).')
            print('The null hypothesis that the population mean difference score is zero can be rejected.')
            print('The conclusion is that the population mean for the treatment/experiment condition is higher than the population mean for the control condition.') 
        else:
            print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')





def main():
    n1 = np.array([10,12,13,14,15,13,14,13,12])    # n1 = 5 * np.random.randint(20, size=40)
    n2 = np.array([1,2,3,4,5,3,4,3,2])     # n2 = 5 * np.random.randint(5, size=40)

    t_teszt = t_test()
    t_teszt.one_sample(n1,n2)


    n1 = np.array(np.random.randint(low=5,high=15, size=30))    # n1 = 5 * np.random.randint(20, size=40)
    n2 = np.array(np.random.randint(low=10, high=15, size=30))     # n2 = 5 * np.random.randint(5, size=40)

    t_teszt = t_test()
    t_teszt.indenpendent_sample(n1,n2)

if __name__ == "__main__":
    main() 