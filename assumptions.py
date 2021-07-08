#!/usr/bin/env python

import scipy.stats
import numpy as np
import pandas as pd
import logging
import unittest


# TODO
# error handling
# variable types
# unittests

def simple_hist(x):
    # legyen egy histogram és a normal eloszlás pdf-je
    x = pd.DataFrame(x).hist()
    plt.savefig('normality_test'.png)

def q_q(self):
    pass

def box_whiskers():
    pass

def kurtosis(x):
    return pd.Series(x).kurtosis()

def skew(x):
    return pd.Series(x).skew()

def normality_test(x: np.ndarray, method=shapiro-wilk) -> tuple:

    df = len(x) -1

    if method == 'shapiro-wilk':
        w,p = scipy.stats.shapiro(x) #  test stat, p-value
        if p >= 0.05:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. ")
        return (w,p)

    elif method == 'kolmogorov-szmirnov':
        k,p = scipy.stats.kstest(x, 'norm')
        if p >= 0.05:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skew} and kurtosis is {kurt}. You can find the histogram of the variable in this folder. ")
    
        return (k,p)
    else:
        raise VariableError 

def homogeneity_of_variance(data):
    f,p = ss.levene(*data)
    #df_b = 
    #df_w =
        if p >= 0.05:
            print(f'The homogeneity of variance assumption was not violated (F({df_b},{df_w})= {f}, p={p}).')
        else:
            print(f' The homogeneity of variance assumption was violated (F({df_b},{df_w})= {f}, p={p}).')

    return f,p



if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.normal(loc=6, scale=5, size=30)
    y = np.random.normal(loc=5, scale=3, size=30)


