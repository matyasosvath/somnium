#!/usr/bin/env python

from scipy.stats import f_oneway
import numpy as np 
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import scipy.stats as ss

import vizualizacio as viz # Adat vizualizáció


#TODO - add assumptions decorator

assmp = dict()

assmp['Shapiro-Wilk'] = {'Test Stat': 4.55, 'P-value': 0.04}
assmp['Levene F-test'] = {'Test Stat': 4.55, 'P-value': 0.04}
assmp['Data Type'] = 'Continuous'
assmp['Repeated'] = False


def anova_test(df, groups: str, scores: str, assumptions: dict, effect_size='eta-square'):
    unique_group_values = list(df[groups].unique())
    
    # Sima np.array-ek kinyerése loop-al, a scipy-nak az kell
    data = []
    for i in unique_group_values:
        data.append(df[df[groups] == i][scores].values)
    
    # One-way ANOVA
    f,p = f_oneway(*data)
    
    # Data visualization
    # Box Chart
    viz.oszlopdiagram(df, groups, scores)
    
    #print(f,p)
    #print(assumptions)
    print(f'A one-way anova test was conducted to test the null hypothesis.')
    
    # Assumptions
    if assumptions['Shapiro-Wilk']['P-value'] > 0.05 and assumptions['Levene F-test']['P-value'] > 0.05:
        print(f'One-way ANOVA assumptions (normality and homogenity of variance) was not violated.')
        
        if p >0.05: 
            print(f'There were no statistically significant differences between group means (ez nem biztos hogy helyes) as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p})").')
        else: 
            print(f'There was a statistically significant difference between groups as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p}).')
            
            #Tukey HSD test
            tukey = pg.pairwise_tukey(df, dv=scores, between= groups, effsize=effect_size)
            if (tukey['p-tukey'].values <= 0.05).any():
                for p in tukey['p-tukey'].values:
                    print(p)
                    t = 'A Tukey post hoc test revealed that'
                    if p > 0.05: 
                        group_a_name  =tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][0]
                        group_b_name  =tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][1]
                        print(f"{t} there was no statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}).")
                    elif p < 0.05: 
                        group_a_name = tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][0]
                        group_b_name = tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][1]
                        ef = tukey[tukey['p-tukey'] == p][effect_size].values[0] # effect-size
                        #print(ef)
                        print(f"{t} there was statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}), with the effect size of Eta-square/Hedges-G: {np.round(ef,2)} ")
                    else:
                        raise ValueError('There is something wring with Tukey Post hoc test')
    elif assumptions['Shapiro-Wilk']['P-value'] <= 0.05 and assumptions['Levene F-test']['P-value'] <= 0.05:
        # Kruskal-Wallis goes here
        print(f'Both Assumptions of Normality and Homogenity of Variance is violated.')
        h,p = ss.kruskal(*data)
        rangatlagok = mean_rank(df, groups=groups, scores=scores)
        if p > 0.05:
            print(f"A Kru")
        else:
            print(f"A Kruskal-Wallis H test showed that there was a statistically significant difference in scores between the different groups χ2(2) = {h}, p = {p}")
            print(f"..with a mean rank score of {rangatlagok[0][0]} of {rangatlagok[0][1]}")
            #TODO: Post hoc Test for kruskal-wallis goes here
        return (h,p)
        
    elif assumptions['Shapiro-Wilk']['P-value'] <= 0.05:
        #TODO
        raise ValueError('The Assumptions of Normality is violated. Please Check if the data is transformable to normal distributions.')
        
    elif assumptions['Levene F-test']['P-value'] <= 0.05:
        #TODO
        raise ValueError('The Assumptions of Homogenity of Variance is violated. Please Check if the data is transformable to normal distributions.')
        
    else:
        raise ValueError("Ide nem kellene jönnöd")
    
    return (f,p)



def mean_rank(df, groups: str, scores: str) -> dict:
  """
  Visszadja a rang átlagokat a különböző csoportoknak a táblázatban, 
  A Kruskal Wallis teszthez kell.
  """
  csoportok_nevei = list(df['species'].unique())
  mean_ranks = []
  for n in csoportok_nevei:
    mean_ranks.append(df[df['species'] == n]['body_mass_g'].rank().mean())

  return list(zip(csoportok_nevei, mean_ranks))







if __name__ == '__main__':
    df = pg.read_dataset('penguins')
    print(df.head())
    anova_test(df,'species', 'body_mass_g', assumptions=assmp)





