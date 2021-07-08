#!/usr/bin/env python

import scipy.stats
import numpy as np
import pandas as pd
import logging
import unittest


def box_whiskers():
    pass

def kurtosis():
    pass

def skew():
    pass



def normality_test(x, method=shapiro-wilk):

    x = pd.DataFrame(x).hist()
    plt.savefig('normality_test'.png)


    # boxs & whiskers
    kurt = x.kurtosis()
    skew = x.skew()
    df = len(x) -1

    if method == 'shapiro-wilk':
        w,p = scipy.stats.shapiro(x) #  test stat, p-value
        if p >= 0.05:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. ")

    elif method == 'kolmogorov-szmirnov':
        k,p = scipy.stats.kstest(arg, method)
        if p >= 0.05:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder. ")
    
    else:
        raise VariableError 

def homogeneity_of_variance(data):
    f,p = ss.levene(*data)
    df_b = 
    df_w =
        if p >= 0.05:
            print(f'The homogeneity of variance assumption was not violated (F({df_b},{df_w})= {f}, p={p}).')
        else:
            print(f' The homogeneity of variance assumption was violated (F({df_b},{df_w})= {f}, p={p}).')

    return f,p





if __name__ == '__main__':
    pass



# Check for normality
class Normality:
    def __init__(self):
        self.shapiro = shapiro
        self.kstest = kstest
        # self.box = box_plot


    def q_q(self):
        pass

    def skewness(self):
        pass

    def kurtosis(self):
        pass

    def shapiro_wilk(self, *args: np.ndarray) -> tuple:

        # another method 1
        data = args[0]
        if args[1]:
            data = args[0] - args[1]
        
        # method 2
        # try:
        #     var1, var2 = data
        #     data = var1 - var2
        # except ValueError:
        #     pass
        return self.shapiro(data)

    def kolmogorov_smirnov(self, *data: np.ndarray) -> tuple:
        try:
            var1, var2 = data
            data = var1 - var2 # one sample t-test
        except ValueError:
            pass
        return self.kstest(data, 'norm')

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.normal(loc=6, scale=5, size=30)
    data2 = np.random.normal(loc=5, scale=3, size=30)

    n = Normality()
    statistic, pvalue = n.shapiro_wilk(data, data2)
    print(statistic, pvalue)
