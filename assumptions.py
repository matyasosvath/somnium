#!/usr/bin/env python

import scipy.stats

def normality_test(x, method=shapiro-wilk):

    x = pd.DataFrame(x).hist()
    kurt = x.kurtosis()
    skew = x.skew()

    print(f'The normality of {x} scores was assessed. The Shapiro-Wilk tests 
    indicated that the scores were  variable skew is {skew} and kurtosis is {kurt}. ')
    if method == 'shapiro-wilk':
        return scipy.stats.shapiro(arg) # test stat, p-value
    
    if method == 'kolmogorov-szmirnov':
        return scipy.stats.kstest(arg, method)


    



