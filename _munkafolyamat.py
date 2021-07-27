#!/usr/bin/env python

from scipy.stats import f_oneway
import numpy as np 
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import scipy.stats as ss

import logging


#import vizualizacio as viz # Adat vizualizáció

logger = logging.getLogger()


#TODO - add assumptions decorator



# def anova_test(df, groups: str, scores: str, assumptions: dict, effect_size='eta-square'):
#     unique_group_values = list(df[groups].unique())
    
#     # Sima np.array-ek kinyerése loop-al, a scipy-nak az kell
#     data = []
#     for i in unique_group_values:
#         data.append(df[df[groups] == i][scores].values)
    
#     # One-way ANOVA
#     f,p = f_oneway(*data)
    
#     # Data visualization
#     # Box Chart
#     viz.oszlopdiagram(df, groups, scores)
    
#     #print(f,p)
#     #print(assumptions)
#     print(f'A one-way anova test was conducted to test the null hypothesis.')
    
#     # Assumptions
#     if assumptions['Shapiro-Wilk']['P-value'] > 0.05 and assumptions['Levene F-test']['P-value'] > 0.05:
#         print(f'One-way ANOVA assumptions (normality and homogenity of variance) was not violated.')
        
#         if p >0.05: 
#             print(f'There were no statistically significant differences between group means (ez nem biztos hogy helyes) as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p})").')
#         else: 
#             print(f'There was a statistically significant difference between groups as determined by one-way ANOVA (F(df_b,df_w) = {f}, p = {p}).')
            
#             #Tukey HSD test
#             tukey = pg.pairwise_tukey(df, dv=scores, between= groups, effsize=effect_size)
#             if (tukey['p-tukey'].values <= 0.05).any():
#                 for p in tukey['p-tukey'].values:
#                     print(p)
#                     t = 'A Tukey post hoc test revealed that'
#                     if p > 0.05: 
#                         group_a_name  =tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][0]
#                         group_b_name  =tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][1]
#                         print(f"{t} there was no statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}).")
#                     elif p < 0.05: 
#                         group_a_name = tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][0]
#                         group_b_name = tukey[tukey['p-tukey'] == p][['A', 'B']].values[0][1]
#                         ef = tukey[tukey['p-tukey'] == p][effect_size].values[0] # effect-size
#                         #print(ef)
#                         print(f"{t} there was statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}), with the effect size of Eta-square/Hedges-G: {np.round(ef,2)} ")
#                     else:
#                         raise ValueError('There is something wring with Tukey Post hoc test')
#     elif assumptions['Shapiro-Wilk']['P-value'] <= 0.05 and assumptions['Levene F-test']['P-value'] <= 0.05:
#         # Kruskal-Wallis
#         print(f'Both Assumptions of Normality and Homogenity of Variance is violated.')
#         h,p = ss.kruskal(*data)
#         rangatlagok = mean_rank(df, groups=groups, scores=scores)
#         if p > 0.05:
#             print(f"A Kruskal-Wallis H test showed that there was no statistically significant difference between groups." )
#         else:
#             print(f"A Kruskal-Wallis H test showed that there was a statistically significant difference in scores between the different groups χ2(2) = {h}, p = {p}")
#             #TODO for loop a rangátlagoknak
#             print(f"..with a mean rank score of {rangatlagok[0][0]} of {rangatlagok[0][1]}")
#             #TODO: Post hoc Test for kruskal-wallis goes here
#             print(f'A Games-Howell post hoc test was conducted for post comparison.')
#             gh = pg.pairwise_gameshowell(data=df, dv=scores, between=scores, effsize='eta-square').round(3)
#             if (gh['pval'].values <= 0.05).any():
#                 for p in gh['pval'].values:
#                     #print(p)
#                     t = 'A Games-Howell post hoc test revealed that'
#                     if p > 0.05:
#                         group_a_name  =gh[gh['pval'] == p][['A', 'B']].values[0][0]
#                         group_b_name  =gh[gh['pval'] == p][['A', 'B']].values[0][1]
#                         print(f"{t} there was no statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}).")
#                     elif p < 0.05:
#                         group_a_name = gh[gh['pval'] == p][['A', 'B']].values[0][0]
#                         group_b_name = gh[gh['pval'] == p][['A', 'B']].values[0][1]
#                         ef = gh[gh['pval'] == p][effect_size].values[0] # effect-size
#                         #print(ef)
#                         print(f"{t} there was statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}), with the effect size of Eta-square/Hedges-G: {np.round(ef,2)} ")
#                     else:
#                         raise ValueError('There is something wring with Tukey Post hoc test')

#         return (h,p)
        
#     elif assumptions['Shapiro-Wilk']['P-value'] <= 0.05:
#         #TODO
#         raise ValueError('The Assumptions of Normality is violated. Please Check if the data is transformable to normal distributions.')
#         # Különben ide is Kruskal-Wallis kell
        
#     elif assumptions['Levene F-test']['P-value'] <= 0.05:
#         #TODO
#         wa = pg.welch_anova(df, dv=scores, between=groups)
#         csoportok, ddof1, ddof2, f, p, np2 = list(wa.values[0])
#         if p < 0.05:
#             print(f"Welch ANOVA test revealed {p} is significant babayyy.")
#         else:
#             print(f"Welch ANO test reveled {p} is not significant.")
#         print("hello there")
#         print(df.isnull().any())
#         df[scores] = df[scores].astype('int')
#         gh = pg.pairwise_gameshowell(data=df, dv=scores, between=scores, effsize='eta-square').round(3)
#         print("hell there again bug")
#         print(gh)
        
#         if (gh['pval'].values <= 1.05).any():
#             for p in gh['pval'].values:
#                 #print(p)
#                 t = 'A Games-Howell post hoc test revealed that'
#                 if p > 0.05:
#                     group_a_name  =gh[gh['pval'] == p][['A', 'B']].values[0][0]
#                     group_b_name  =gh[gh['pval'] == p][['A', 'B']].values[0][1]
#                     print(f"{t} there was no statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}).")
#                 elif p < 0.05:
#                     group_a_name = gh[gh['pval'] == p][['A', 'B']].values[0][0]
#                     group_b_name = gh[gh['pval'] == p][['A', 'B']].values[0][1]
#                     ef = gh[gh['pval'] == p][effect_size].values[0] # effect-size
#                     #print(ef)
#                     print(f"{t} there was statistically significant difference between the {group_a_name} and {group_b_name} groups (p = {np.round(p, 3)}), with the effect size of Eta-square/Hedges-G: {np.round(ef,2)} ")
#                 else:
#                     raise ValueError('There is something wring with Tukey Post hoc test in Wlech ANOVA test branch.')

#         return (f,p)

#         #raise ValueError('The Assumptions of Homogenity of Variance is violated. Please Check if the data is transformable to normal distributions.')
#     else:
#         raise ValueError("Ide nem kellene jönnöd")
    
#     return (f,p)



# def mean_rank(df, groups: str, scores: str) -> dict:
#   """
#   Visszadja a rang átlagokat a különböző csoportoknak a táblázatban, 
#   A Kruskal Wallis teszthez kell.
#   """
#   csoportok_nevei = list(df['species'].unique())
#   mean_ranks = []
#   for n in csoportok_nevei:
#     mean_ranks.append(df[df['species'] == n]['body_mass_g'].rank().mean())

#   return list(zip(csoportok_nevei, mean_ranks))



##############################
######## KORRELÁCIÓ ##########
##############################

class HipotezisTeszt:
    """
    Az összes teszt itt legyen
    """
    def __init__(self) -> None:
        pass


class Assumptions:
    def __init__(self) -> None:
        pass

    def test_assumptions(self, adatok, method):
        print(adatok, method)

class Vizualizacio:
    def __init__(self) -> None:
        pass

class AdatTisztitas:
    def __init__(self) -> None:
        pass



# Base Class a folyamatoknak
# class BaseClass(HipotezisTeszt, Assumptions, Vizualizacio, AdatTisztitas):
#     def __init__(self):
#         pass
    
#     def adatok_kinyerese(self, *args): # hol legyen?
#         pass



class Korrelacio(HipotezisTeszt, Assumptions, Vizualizacio, AdatTisztitas):
    def __init__(self):

        #self.assumptions = dict()
        self.adatok = None # multiple pd.series

        self.eredmenyek = None

        logger.info("Korrelacio class successfully initiated!")

    def adatok_kinyerese(self, df, groups, scores)):
        """
        Adatok kinyerése akár így van megadva: df, groups, scores. Akár így df[['oszlop1', 'oszlop2']].
        Hiszen a drága pszichológusok mindkettő fél képpen elemzik az adatokat.

        ELŐSZÖR EGYIKRE CSINÁLD MEG!!!!

        Returns: multiple pd.series.
        """
        # Sima np.array-ek kinyerése loop-al, a scipy-nak az kell


        unique_group_values = list(df[groups].unique())
        self.adatok = []
        for i in unique_group_values:
            self.adatok.append(df[df[groups] == i][scores].values)
        
        # do some func()

        return self.adatok

    def run(self, df, groups, scores):
        self.adatok = self.adatok_kinyerese(df, groups, scores)

        self.assumptions = self.test_assumptions(self.adatok, method="correlation")
        # egyik funkciója beleírni a .txt-ba, másik paramterátadás

        self.scatter_plot(self.adatok) # csak egy kép mentése a scatter_plotokról, q-q plotokról

        if self.assumptions is not None:
            if self.assumptions['Bivariate Normality']: # no outliers and normal distribution
                self.pearson_korrelacio(self.adatok)
            
            elif self.assumptions['Bivariate Normality'] == False:

                self.spearman_korrelacio(self.adatok)

                if self.assumptions["Univariate Outliers"]:
                    self.biweight_midcorrelation(self.adatok)
                    self.percentage_bend_correlation(self.adatok)
                    
                elif self.assumptions["Bivariate Outliers"]:
                    self.skipped_correlation(self.adatok)
                    self.shepherd_correlation(self.adatok)                    
                    
                else:
                    raise ValueError("Something wrong with Correlacio class, self.assumptions bivariate/univariate outliers")

            else:
                raise ValueError("Something wrong with Correlacio class, self.assumptions.")

                        



# először a logika
# nem kell minden egyszerre mert overwhelming
        
        

























#if __name__ == '__main__':

    
import unittest



assmp = dict()
    
assmp['Shapiro-Wilk'] = {'Test Stat': 4.55, 'P-value': 0.02}
assmp['Levene F-test'] = {'Test Stat': 4.55, 'P-value': 0.04}
assmp['Data Type'] = 'Continuous'
assmp['Repeated'] = False

class TestAnova(unittest.TestCase):
    # test function to test equality of two value



    def test_one_way_anova(self):
        
        df = pg.read_dataset('penguins')
        observed= anova_test(df, 'species','body_mass_g', assumptions=assmp)
        expected = 0.05
        # error message in case if test case got failed
        message = "Message amit kiprintel"
        # assertEqual() to check equality of first & second value
        self.assertEqual(firstValue, secondValue, message)    

import unittest

def add_fish_to_aquarium(fish_list):
    if len(fish_list) > 10:
        raise ValueError("A maximum of 10 fish can be added to the aquarium")
    return {"tank_a": fish_list}


class TestAddFishToAquarium(unittest.TestCase):
    def test_add_fish_to_aquarium_success(self):
        actual = add_fish_to_aquarium(fish_list=["shark", "tuna"])
        expected = {"tank_a": ["shark", "tuna"]}
        self.assertEqual(actual, expected)
    
    
    #df = pg.read_dataset('penguins')
    #print(df.head())

    assmp = dict()
    
    assmp['Shapiro-Wilk'] = {'Test Stat': 4.55, 'P-value': 0.02}
    assmp['Levene F-test'] = {'Test Stat': 4.55, 'P-value': 0.04}
    assmp['Data Type'] = 'Continuous'
    assmp['Repeated'] = False


    #df.dropna(inplace=True)
    #df['body_mass_g'].astype('int')
    #anova_test(df,'species', 'body_mass_g', assumptions=assmp)





