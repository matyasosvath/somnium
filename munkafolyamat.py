#!/usr/bin/env python

from math import exp
import numpy as np 
import pandas as pd
import pingouin as pg
import scipy.stats as ss
import itertools

import logging

logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(stream_handler)


# Saját modulok
from _vizualizacio import *
from tesztek import *
from adattisztitas import *


##############################
######## ASSUMPTIONS #########
##############################


class Assumptions:
    def __init__(self) -> None:
        pass

    logger.info("Assumptions successfully initialized")


    def test_for_assumptions(self, x, method):
        assmp = dict()
        
        if method == "correlation":
            # Mi kell ide?
            # normality plot (kész)
            # normalitás teszt (kész)
            # outliers (kész a multivariate normality test miatt)
        
            # Tests
            assmp = normality_test(x, method='multivariate')
            # assmp['Outliers'] = check_for_multivariate_outliers(df = self.data, data=self.data[[self.x,self.y]])
            # print(self.assumptions)
            return assmp

        elif method == "two groups":
            pass
      
        elif method == "three or more groups":
            pass
      
        elif method == "regression":
            pass

        elif method == "categorical":
            pass

        else:
            raise ValueError


##############################
######## KORRELÁCIÓ ##########
##############################


class Korrelacio(Assumptions, HipotezisTesztek):
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.data = pd.concat([x,y], axis=1) # két külön series-ből egy datafram legyen a "multivariate test" kedvéért
        # kell hogy a x.name-nek neve legyen különben hibát jelez ki
      

        self.x_adattipus = adattipus(self.x)
        self.y_adattipus = adattipus(self.y)
        #print(self.x_adattipus, self.y_adattipus)

        self.assumptions = self.test_for_assumptions(self.data, method="correlation")
        self.assumptions['Outliers'] = check_for_multivariate_outliers(self.data)

        logger.info("Korrelacio class successfully initialized")
    
    def run(self):  

        # Remove outliers
        self.removed_outliers = remove_multivariate_outliers(self.data)

        #print(f"removed outliers az ez: {self.removed_outliers}")
        
        logger.info("Outliers successfully removed!")
        
        self.x, self.y = self.removed_outliers.iloc[:, 0], self.removed_outliers.iloc[:, 1]
        # print(type(self.x), type(self.y))
        # print(self.x)
        # print(self.y)

        # Folytonos-folytonos
        if self.y_adattipus == 'ratio-interval' and self.x_adattipus == 'ratio-interval':
            
            # Vizualizacio
            scatter_plot(df, x=self.x,y=self.y)
        
            if self.assumptions['Normality Test']['Henze-Zirkler']['P-value'] > 0.05 and self.assumptions['Outliers']['Multivariate Outliers'] == False:

                self.pearson(self.x,self.y)
                logger.info("Pearson test successfully run!")

            elif self.assumptions['Normality Test']['Henze-Zirkler']['P-value'] <= 0.05 or self.assumptions['Outliers']['Multivariate Outliers']:

                #print(self.x.ndim, self.y.ndim)
                self.spearman(self.x,self.y)#if self.assumptions["Univariate Outliers"]:
                logger.info("Spearman test successfully run!")

                # Bivariate Outliers
                if self.assumptions["Outliers"]['Multivariate Outliers']: 
                    self.skipped_spearman_correlation(self.x,self.y)
                    self.shepherd_pi_correlation(self.x,self.y)
                    logger.info("Skipped spearman and shepherd pi corr test ran successfully!")         
                
                # Univariate outliers
                # else:
                #     self.biweight_correlation(self.x,self.y)    
                #     self.percentage_bend_correlation(self.x,self.y)

            else:
                raise ValueError
        
        # Folytonos-Ordinális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'ratio-interval':
            # Spearman rankkorrelacio
            self.spearman(self.x,self.y)
            logger.info("Spearman test successfully run!")

        # Folytonos-Nominális
        elif self.y_adattipus == 'ratio-interval' and self.x_adattipus == 'ordinal':
            # Point-biserial correlation coefficient
            self.point_biserial_correlation(self.x,self.y)
            logger.info("Point biserial correlation test successfully run!")
            
        # Ordinális-Ordinális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'ordinal':
            # Kendalls rank correlation coefficient
            self.kendall_tau(self.x,self.y)
            logger.info("Kendall tau-b test successfully run!")

        # Ordinális-Nominális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'nominal':
            # Rank-biserial correlation coefficient
            self.rank_biserial_correlation(self.x,self.y)
            logger.error("Rank biserial correlation are not implemented yet!")

        # Nominális-nominális
        elif self.y_adattipus == 'nominal' and self.x_adattipus == 'nominal':
            # Phi coefficient or Matthews correlation coefficient 
            self.phi_coeff_matthews_coeff(self.x,self.y)
            logger.info("Matthews coefficient test successfully run!")

        else:
            raise ValueError("Hiba. Nincs tobb korrelacio lehetoseg.")




if __name__ == '__main__':
    import pingouin as pg
    #df = pg.read_dataset('penguins')


    # AdatTisztitas
    df.dropna(inplace=True)

    data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
        'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
        'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
        }
    #a = Assumptions()

    df = pd.DataFrame(data,columns=['score', 'hours', 'prep','grade'])

    for i,j in list(itertools.combinations(list(df), 2)):
        print(i,j)
        #print(type(i), type(j))
        #print(df[[i,j]])

        korr = Korrelacio(df[i], df[j])
        korr.run()

        korr = Korrelacio(df[j], df[i])
        korr.run()
    
    #korr = Korrelacio(df['grade'], df['score'])
    #korr.assumptions
    #korr.run()







############# KORÁBBI ELEMZÉSEK




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


