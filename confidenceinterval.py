#!/usr/bin/env python

from scipy.stats import t
import numpy as np


# FIX
# Unequal sample size automate & calculate sample size
# CI for correlation, proportion
# Integrate 2 function

def interval(*variables, ci=0.95, compute = 'mean', sample_size='equal', pop_var = 'unknown'):

    n1 = variables[0]
    try:
        n2 = variables[1]
    except (ValueError, IndexError):
        pass

    # If the pop standard deviation is known
    if pop_var == 'known':
        n = len(n1)
        sem = std/np.sqrt(n) 
        if ci == 0.95:
            lower = x.mean() - 1.96* sem
            upper = x.mean() + 1.96* sem
            return tuple(lower, upper)
        if ci == 0.90: # or 99
            pass
    # If the pop standard deviation is unknown
    else:    
        if compute == 'mean':
            n = len(n1)
            m = n1.mean()

            sem = (n1.std()/np.sqrt(n))
            dof = n-1

            t_cl = t.interval(ci, dof) # return an interval
            t_cl = abs(t_cl[1]) # need one value
            print(f't_cl is {t_cl}; sem is {sem}')

            lower = m - t_cl * sem
            upper = m + t_cl * sem
            return tuple((lower, upper))

        if compute == 'difference between means':

            n = len(n1)
            # Compute means
            m1 = n1.mean()
            m2 = n2.mean()

            # Compute standard error of the difference between means
            MSE = n1.std()- n2.std()/2
            sem_m1_m2 = np.sqrt((2*MSE)/n)

            # Degrees of freedom
            dof = len(n1)-1 + len(n2) - 1

            # t-distribution
            t_cl = t.interval(ci, dof) # return an interval
            t_cl = abs(t_cl[1]) # need one value
            # print(f't_cl is {t_cl}; sem is {sem}')

            m_diff = m1-m2
            lower = m_diff - t_cl * sem_m1_m2
            upper = m_diff + t_cl * sem_m1_m2
            return tuple((lower, upper))
        if compute == 'corr':
            pass
        if compute == 'proportion':
            pass
    



def interval(*variables, ci=95, method='bootstrap', iters=1000):
    x,y = variables
    means = sorted(list((np.mean(np.random.choice(x, size=len(x), replace=True)) for i in range(iters))))
    # for i in range(iters):
    #     x = np.random.choice(x, size=len(x), replace=True)
    #     means.append(x.mean())
    

    # Trim endpoints of resampled CI
    trim = ((1 - (ci/100))/2)
    endpoints = int(trim*1000)
    trimmed_ci = means[endpoints:-endpoints]
    lower, upper = min(trimmed_ci), max(trimmed_ci)

    print(trim, endpoints, trimmed_ci, lower, upper)

    # or 
    trimmed_ci = means[25:]

    return lower, upper



if __name__ == '__main__':
    np.random.seed(42)
    n1 = np.random.randint(3,10, size=25)
    n2 = np.random.randint(5,10, size=10)


    lower, upper = interval(n1,n2, ci=95)
    print(f'CI Lower {lower}, Upper {upper}, Mean: {n1.mean()}')