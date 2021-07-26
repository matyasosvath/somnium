#!/usr/bin/env python

import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import unittest
import pingouin as pg

# TODO
# error handling
# variable types
# unittests

def variable_type(x):
    pass


def simple_hist(x):
    # legyen egy histogram es a normal eloszlas pdf-je
    x = pd.DataFrame(x).hist()
    plt.savefig('normality_test'.png)



def box_whiskers():
    pass


def kurtosis(x):
    return pd.Series(x).kurtosis()


def skew(x):
    return pd.Series(x).skew()

def normality_test(x, method='shapiro-wilk'):
    """

    Returns tuple (test statistic, p-value)
    """
    df = len(x) - 1

    krts = kurtosis(x)
    skw = skew(x)


    if method == 'shapiro-wilk':
        w, p = np.round(ss.shapiro(x),3)  # test stat, p-value
        print('ez jo')
        if p >= 0.05:
            print(f"The normality of {x.name} (majd oszlopnevet behelyettesiteni) scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.")
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. ")       

        return (w, p)

    elif method == 'kolmogorov-szmirnov':
        k, p = np.round(ss.kstest(x, 'norm'),3)
        if p >= 0.05:
            print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were normally distributed (W({df}))={k}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were not normally distributed (W({df}))={k}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder. ")
        return (k, p)
    else:
        raise ValueError('Only Shapiro-Wilk and Kolmogorov-Szmirnov are optional')


def homogeneity_of_variance(*data):
    """

    Returns tuple (test statistic, p-value)
    """
    f, p = np.round(ss.levene(*data), 3)
    df_b = 0
    df_w = 0
    # Pingouin módszere, df, scores, groups
    #pg.homoscedasticity(data=df_anova, dv=scores, group=groups)

    if p >= 0.05:
        print(f"Levene's test for equality of variances is not significant (F({df_b},{df_w})= {f}, p={p}).")
    else:
        print(f' The homogeneity of variance assumption was violated (F({df_b},{df_w})= {f}, p={p}).')

    return f, p




def normality_test_decorator(x, method='shapiro-wilk'):
    """
    Maint to by executed as a decorator
    Executed a separate function returns tuple (test statistic, p-value)

    """
    df = len(x) - 1

    krts = kurtosis(x)
    skw = skew(x)


    if method == 'shapiro-wilk':
        w, p = np.round(ss.shapiro(x),3)  # test stat, p-value
        print('ez jo')
        if p >= 0.05:
            print(f"The normality of {x.name} (majd oszlopnevet behelyettesiteni) scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.")
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. ")
        
        assmp = dict()
        assmp['Shapiro-Wilk Test Statistic'] = w
        assmp['p-value'] = p         

        return (w, p)

    elif method == 'kolmogorov-szmirnov':
        k, p = np.round(ss.kstest(x, 'norm'),3)
        if p >= 0.05:
            print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were normally distributed (W({df}))={k}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
        else:
            print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were not normally distributed (W({df}))={k}, p={p}.')
            print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder. ")
        return (k, p)
    else:
        raise ValueError('Only Shapiro-Wilk and Kolmogorov-Szmirnov are optional')




if __name__ == '__main__':
    import unittest

    np.random.seed(42)
    x = pd.Series(np.random.normal(loc=6, scale=5, size=30), name='oszlop_nev1')
    y = pd.Series(np.random.normal(loc=10, scale=3, size=30), name='oszlop_nev2')
    z = pd.Series(np.random.normal(loc=5, scale=4, size=30), name='oszlop_nev3')

    print(normality_test(x))
    print(normality_test(x, method="kolmogorov-szmirnov"))

    class TestAssumptions(unittest.TestCase):
        def test_normality_test(self):
            actual = normality_test(x, method='shapiro-wilk')
            expected = (0.9751381874084473, 0.6868011355400085)
            self.assertEqual(actual, expected)

        def test_homogeneity_of_variance_1(self):
            actual = homogeneity_of_variance(x, y, z)
            self.assertEqual(
                actual, (1.914, 0.154), "OK with homogeneity of variance test!")

        def test_homogeneity_of_variance_2(self):
            actual = homogeneity_of_variance(x, y)
            self.assertEqual(
                actual, (3.952, 0.052), "OK homogeneity of variance test!")
