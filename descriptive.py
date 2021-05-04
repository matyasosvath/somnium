#!/usr/bin/env python

import pandas as pd
import numpy as np

def mean(scores): # first moment
    return sum(scores)/len(scores)

def deviation(scores):
    # Átlagos eltérés: az egyes adatok számtani átlagtól való abszolút eltéréseinek átlaga.
    return np.round(sum(abs(x - mean(scores)) for x in scores) / (len(scores)-1))

def covariance(x,y):
    # Deviations from mean
    d_x = x - x.mean()
    d_y = y - y.mean()

    cov = np.dot(d_x,d_y)/x.shape[0] 
    return cov


def std(scores): 
    return 

def var(scores): # second moment
    pass


def percentilerank(scores, your_score):
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1
    percentile_rank = 100.0 * count / len(scores)
    return percentile_rank


def z_score(scores):
    return (scores - scores.mean())/scores.std()


def factorial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)


def permutation(x):
    pass



class EffectSize:
    def __init__(self):
        pass


    def prop(self, table: [list, np.ndarray, pd.core.series.Series], method='arr') -> int:
        """
        Risk of a particular event (e.g heart attack) happening.

        Input 
            2x2 matrix/table (pandas, numpy)
            # List: [[90,10],[79,21]] (Not implemented)

                | Positive  | Negative
        Group A |     90    |  10
        Group B |     79    |  21

        Returns
            One of the called method results.
        """

        def arr():
            """
            Absolute Risk Reduction 
    
            C - the proportion of people in the control group with the ailment of interest (illness)  
            T - the proportion in the treatment group.

            Then, ARR = C-T
            """

            C = table.iloc[0,1]/table.iloc[0].sum() # Negative Outcome / Sum (Group 1)
            T = table.iloc[1,1]/table.iloc[1].sum()  # Negative Outcome / Sum (Group 2)
            return np.round(abs(C-T), 3)

        def rrr():
            """
            Relative Risk Reduction (RRR)
            
            Measure the difference in terms of percentages.


            """
            return (C-T)/C * 100

        def odds_ratio():
            pass


        def fourth_measure():
            """
            4th Measure:
            The number of people who need to be treated in order to
            prevent one person from having the ailment of interest.
            """
            return 1/arr()

        if method == 'arr':
            return arr()

        if method == 'rrr':
            return rrr()

        if method == 'odds_ratio':
            return odds_ratio()

        if method == 'fourth_measure':
            return fourth_measure()

    def diff_means(self):
        pass
 



if __name__ == '__main__':


    scores = [11,32,23,54,65]
    print(mean(scores))

    print(deviation(scores))


    table = pd.DataFrame([[90,10],[79,21]], index = ['A', 'B'], columns= ['plus', 'minus'])
    print(table)

    ef = EffectSize()
    ef_prop = ef.prop(table,method='arr')
    print(ef_prop)
