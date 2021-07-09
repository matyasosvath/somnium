#!/usr/bin/env python


from scipy.stats import f_oneway
import numpy as np 
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

import assumptions as ass


class Anova: #MultipleGroups?
    def __init__(self, data, assumptions):
        self.data =  data
        self.assumptions = assumptions #TODO

    def anova_one_way(self):
        f,p = f_oneway(self.data)
        if p >0.05:
            print(f'There were no statistically significant differences between group means as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p})").')
        elif p <= 0.05:

            print(f'There was a statistically significant difference between groups as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p}). \
                    A Tukey post hoc test revealed that the time to complete the problem was statistically significantly lower after taking the intermediate (23.6 ± 3.3 min, p = .046) and advanced (23.4 ± 3.2 min, p = .034) course compared to the beginners course (27.2 ± 3.0 min). There was no statistically significant difference between the intermediate and advanced groups (p = .989).')

    # if elf.assumptions['homogeneity_of_variance'] == False
    def welch_f_test(self): # Brown and Forsythe test,  Kruskal-Wallis H test
        pass

    def post_hoc(self, groups, scores):
        if self.assumptions['homogeneity_of_variance'] == True:
        #if homogeneity_of_variance == True:
            tukey = pg.pairwise_tukey(data=df, 
                                    dv='score',
                                    between='group', 
                                    effsize='eta-square').round(3)
            print(tukey)
            print(f'A Tukey post-hoc test revealed that the time to complete the problem was statistically significantly \
                    lower after taking the intermediate (23.6 ± 3.3 min, p = .046) and advanced (23.4 ± 3.2 min, p = .034) course compared to the beginners course (27.2 ± 3.0 min). \
                    There was no statistically significant difference between the intermediate and advanced groups (p = .989).')
        else:
            gh = pg.pairwise_gameshowell(data=df, 
                                        dv='score', # depenedent var
                                        between='group', # independent groups
                                        effsize='eta-square').round(3) # effect size
            print(gh)
            print(f'A Games-Howell post-hoc test revealed that the time to complete the problem was statistically significantly \
                    lower after taking the intermediate (23.6 ± 3.3 min, p = .046) and advanced (23.4 ± 3.2 min, p = .034) course compared to the beginners course (27.2 ± 3.0 min). \
                    There was no statistically significant difference between the intermediate and advanced groups (p = .989).')







if __name__ == '__main__':
    df = pd.DataFrame({'score': [85, 86, 88, 75, 78, 94, 98, 79, 71, 80,
                             91, 92, 93, 90, 97, 94, 82, 88, 95, 96,
                             79, 78, 88, 94, 92, 85, 83, 85, 82, 81],
                   'group': np.repeat(['a', 'b', 'c'], repeats=10)}) 
    print(df)


