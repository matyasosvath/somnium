#!/usr/bin/env python

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as ss
from sklearn import metrics

#from sklearn.utils import shuffle


import logging


logger = logging.getLogger()


# Saját moduleok
from _vizualizacio import *



##############################
### ADAT TÍPUSOK ############
##############################


def adattipus(oszlop):
    "oszlop: pd.Series or np.array"
    a = len(oszlop.unique().tolist())
    if a <= 2:
        return 'nominal'
    if 2 < a <= 8:
        return 'ordinal'
    if a > 8:
        return 'ratio-interval'

""" 
def adattipuskkezelo(adat):
    adattipusok = {}
    for col in adat.columns:
        a = len(adat[col].unique().tolist())
        if a <= 2:
            adattipusok[col] = 'nominal'
        if 2 < a <= 8:
            adattipusok[col] = 'ordinal'
        if a > 8:
            adattipusok[col] = 'ratio-or-interval'
    #print(f'{adattipusok} \n')
    with open('jellemzok.json', 'w') as f:
        json.dump(adattipusok, f)
    return adattipusok
 """

#######################
### HIPOTEZIS TESZT ###
#######################


class HipotezisTesztek:

    def __init__(self):
        pass

    logger.info("Hipotezis tesztek successfully initialized")
    
    # Folytonos-folytonos
    def pearson(self,x,y):
        pearson = pg.corr(x, y, tail='two-sided', method='pearson').round(3)
        n,r,ci95,p,bf10, power =  pearson.values[0]

        pos_neg = 'postively' if r>0 else 'negatively'

        print(f'The relationship between {x.name} and {y.name} was assessed.')
        
        if p <= 0.05:
            print(f"A Pearson correlation test showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"A Pearson correlation test showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95

    
    def spearman(self,x,y):
        """
        
        Also possible to use this: df['score'].corr(df['grade']) for instance
        """
        spearman = pg.corr(x, y, tail='two-sided', method='spearman').round(3)
        n, r,ci95,p, power = spearman.values[0]
        
        pos_neg = 'postively' if float(r)>0 else 'negatively'

        # Report
        print(f'The relationship between {x.name} and {y.name} was assessed.')
        
        if p <= 0.05:
            print(f"Spearman correlation test showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"Spearman correlation test showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95
        #return pg.corr(x, y, tail='two-sided', method='spearman').round(3)
    def biweight_correlation(self, x,y):
        """
        Biweight midcorrelation (robust)

        """
        
        biweight = pg.corr(x, y, method="bicor").round(3)
        n, r,ci95,p, power = biweight.values[0]

  
        pos_neg = 'postively' if float(r)>0 else 'negatively'

        # Report
        print(f"The relationship between {x.name} and {y.name} was assessed with biweight midcorrelation (robust).")

        if p <= 0.05:
            print(f"Biweight midcorrelation (robust) showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"Biweight midcorrelation (robust) showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95


    def percentage_bend_correlation(self,x,y):
        """
        Percentage bend correlation (robust)

        """
        
        perc_bend = pg.corr(x, y, method="bicor").round(3)
        n, r,ci95,p, power = perc_bend.values[0]



        pos_neg = 'postively' if float(r)>0 else 'negatively'

        # Report
        print(f"The relationship between {x.name} and {y.name} was assessed with Percentage bend correlation (robust).")

        if p <= 0.05:
            print(f"Percentage bend correlation (robust) showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"Percentage bend correlation (robust) showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95


    def shepherd_pi_correlation(self,x,y):
        """
        Shepherd’s pi correlation (robust)
        """
        
        shepherd = pg.corr(x, y, method="shepherd").round(3)
        n, outliers, r,ci95,p, power = shepherd.values[0]


        pos_neg = 'postively' if float(r)>0 else 'negatively'

        # Report
        print(f"The relationship between {x.name} and {y.name} was assessed with Shepherd’s pi correlation (")

        if p <= 0.05:
            print(f"Shepherd’s pi correlation showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"Shepherd’s pi correlation (robust) showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95


    def skipped_spearman_correlation(self,x,y):
        """
        Skipped spearman correlation (robust)
        """
        
        skipped =  pg.corr(x, y, method="skipped").round(3)
        n, outliers, r,ci95,p, power = skipped.values[0]

        #print(r)
        #print(type(r))
        pos_neg = 'postively' if float(r)>0 else 'negatively'

        # Report
        print(f"The relationship between {x.name} and {y.name} was assessed with Skipped spearman correlation.")

        if p <= 0.05:
            print(f"Skipped spearman correlation test showed that scores among the group of {x.name} and the groupd of {y.name} were {pos_neg} correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")
        else:
            print(f"Skipped spearman correlation showed that scores among the group of {x.name} and the groupd of {y.name} were not correlated, r(df)= {r}, p={p} (CI 95%: {ci95}), N={n} ")

        return r,p, ci95

    def kendall_tau(self, x,y):
        tau_b =  pg.corr(x, y, method="kendall").round(3)
        n, r,ci95,p, power = tau_b.values[0]
        return r, ci95, p

    def point_biserial_correlation(self, x,y):
        test_stat, p = np.round(ss.pointbiserialr(x, y),4)
        return test_stat, p

    def rank_biserial_correlation(self, x,y):
        raise NotImplementedError
    
    def phi_coeff_matthews_coeff(self, x,y):
        """
        Phi Coefficient/Matthews Correlation Coefficient

        #TODO CI, p value missing, implement manually
        """
        r =  metrics.matthews_corrcoef(x,y)
        p = pvalue(x,y, metrics.matthews_corrcoef)
        #ci = ci()
        return r, p



from sklearn.utils import shuffle
import numpy as np

def permutation(v1,v2):
  #print('permut')
  v1 = pd.Series(v1)
  v2 = pd.Series(v2)
  data = pd.concat([v1,v2])
  data = shuffle(data)
                
  v1 = resample(data, size=len(v1), replace=False)
  v2 = resample(data, size=len(v2), replace=False)
  return v1, v2

def resample(x, size, replace = False):
  return np.random.choice(x, size=size, replace=replace)

def pvalue(x,y, hypothesis_test, iter = 1000):
  actual = hypothesis_test(x,y)
  #print(actual)
  #permute_dist = [phi_coeff_matthews_coeff(permutation(x,y)) for _ in range(iter)] # return value in permutation(x,y) do not works, dont know why
  
  permute_dist = []
  for _ in range(iter):
    a,b = permutation(x,y)
    permute_dist.append(hypothesis_test(a,b))
  #print(permute_dist)

  count = sum(1 for i in permute_dist if i >= actual)
  return count/iter


### Metrics

def mean(x):
    return x.mean()

def median(x):
    return x.median()

def median_absolute_deviation(x):
    return ss.median_absolute_deviation(x)



def kurtosis(x):
    k = np.round(x.kurtosis(), 3)
    print(f"The {x.name}'s kurtosis is {k}.")     
    return k # pd.Series(x).kurtosis() -> csak egy series-re jó

def skew(x):
    s = np.round(x.skew(), 3) # pd.Series(x).skew() -> csak egy series-re jó
    print(f"The {x.name} skew is {s}.")
    return s

def shapiro_wilk_test(x):
    df = len(x) - 1

    w, p = np.round(ss.shapiro(x),3)  # test stat, p-value

    kurtosis(x)
    skew(x)

    if p >= 0.05:
        print(f"The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.")

    else:
        print(f'The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')

    return (w, p)

def kolmogorov_szmirnov_teszt(x):

    df = len(x) - 1

    k, p = np.round(ss.kstest(x, 'norm'), 3)

    kurtosis(x)
    skew(x)

    if p >= 0.05:
        print(f'The normality of {x.name} scores was assessed. The Kolmogorov-smirnov tests indicated that the scores were normally distributed (W({df}))={k}, p={p}.')
        # You can find the histogram of the variable in this folder.")
    else:
        print(f'The normality of {x.name} scores was assessed. The Kolmogorov-smirnov tests indicated that the scores were not normally distributed (W({df}))={k}, p={p}.')
        # You can find the histogram of the variable in this folder.

    return (k, p)

def multivariate_normality_test(x, alpha=0.05):

    hz_test_stat , p, is_normal = pg.multivariate_normality(x, alpha=alpha)


    df = x.shape[0] - 1

    if p >= 0.05:
        print(f'The normality of {x.columns[0]} and {x.columns[1]} was assessed (test: {list(x.columns)}). Henze-Zirkler multivariate normality test indicated that the scores were normally distributed (W({df}))={hz_test_stat}, p={p}.')
        # You can find the histogram of the variable in this folder.")
    else:
        print(f'The normality of {x.columns[0]} and {x.columns[1]} was assessed (test: {list(x.columns)}). Henze-Zirkler multivariate normality test indicated that the scores were not normally distributed (W({df}))={hz_test_stat}, p={p}.')
        # You can find the histogram of the variable in this folder.


    return (hz_test_stat, p)

def normality_test(x, method='shapiro-wilk'):
    """
    Shapiro-Wilk or Kolmogorov Smirnov test
    Returns tuple (test statistic, p-value)
    """
    #TODO Csináld meg hogy akkor is jó leygen ha toöbb series-t akarok fogadni
    #if x.shape[1] >= 2:
    #  print('hello')
    #  kurtosises = x.kurtosis()
    #  skews = x.skew()

    # Vizualizacio
    try:
        normality_plot(x)
    except Exception:
        pass

    # Tests
    if method == 'shapiro-wilk':
        w,p = shapiro_wilk_test(x)

        norm_test_dict = {'Normality Test': {'Shapiro-Wilk': {'Test Statistic': w, 'P-value': p}}}

        return norm_test_dict

    elif method == 'kolmogorov-szmirnov':
        k,p = kolmogorov_szmirnov_teszt(x)
      
        norm_test_dict = {'Normality Test': {'Kolmogorov-Szmirnov': {'Test Statistic': k, 'P-value': p}}}

        return norm_test_dict

    elif method == 'multivariate':
        hz_test_stat, p = multivariate_normality_test(x)
        
        norm_test_dict = {'Normality Test': {'Henze-Zirkler': {'Test Statistic': hz_test_stat, 'P-value': p}}}

        return norm_test_dict

    else:
        raise ValueError(
            'Only Shapiro-Wilk and Kolmogorov-Szmirnov are optional. Mit keresel itt?')



# OUTLIERS
## UNIVARIATE OUTLIERS

def detect_univariate_outlier_boundary(x):
    """
    Detect univariate outliers on a variable (pd.Series) based on: Median +- 3 MAD
    """
    p = x.median() + 3* ss.median_absolute_deviation(x)
    n = x.median() - 3* ss.median_absolute_deviation(x)
    return  n,p

def count_univariate_outliers(x):
    """
    Return the number of univariate outliers on a variable (pd.Series).
    """
    n,p = detect_univariate_outlier_boundary(x)
    o1 = x[x > p]
    o2 = x[x < n]
    return len(o1 + o2)

def remove_univariate_outliers(x):
    """
    Remove univariate outliers for a variable (pd.Series).
    """
    n,p = detect_univariate_outlier_boundary(x)
    return x[~(x < n) & ~(x > p)]


## MULTIVARIATE OUTLIERS

def check_for_multivariate_outliers(data):
    df_removed = remove_multivariate_outliers(data)

    print(f"Multivariate outliers were removed based on Mahalanobis distance.")
    
    if df_removed.shape[0] != df.shape[0]:
        return {'Multivariate Outliers': True}
    else:
        return {'Multivariate Outliers': False}

def remove_multivariate_outliers(data):
    """
    Remove multivariate outliers from a df, based on Mahalanobis Distance.

    data: dataframe with multiple pd.series

    Képlet: D^2 = (x-m)^T \cdot C^{-1} \cdot (x-m)
    
    Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.

    A p-value that is less than .001 is considered to be an outlier.
    
    Function excludes data (df) where the p-value is less than 0.001.

    Example

    """

    degress_of_freedom = data.shape[1] - 1

    x_mu = data - np.mean(data)

    cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    
    data['mahalanobis'] = mahal.diagonal()
    data['p'] = 1 - ss.chi2.cdf(data['mahalanobis'], degress_of_freedom)
    data = data[data['p'] > 0.001]
    
    return data.drop(['mahalanobis', 'p'], axis=1)



if __name__ == '__main__':
    import pingouin as pg
    df = pg.read_dataset('penguins')
    df.dropna(inplace=True)

    ht = HipotezisTesztek()

    ht.spearman(df['body_mass_g'], df['bill_depth_mm'])
    ht.pearson(df['body_mass_g'], df['bill_depth_mm'])

    normality_test(df['body_mass_g'], method='shapiro-wilk')
    normality_test(df['body_mass_g'], method='kolmogorov-szmirnov')

    # Test removing multivariate outliers
    data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
        'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
        'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
        }
    print(df.head())
    df = pd.DataFrame(data,columns=['score', 'hours', 'prep','grade'])
    print(df.head())
















####################### KORÁBBI ELEMZÉSEK


# class PermutationTest:
#         def __init__(self, *data):
#                 self.data = data
#                 self.actual = self.test_stat(data)

        # def permutation(self):
        #         v1, v2 = self.data[0], self.data[1]

        #         # for i in self.data[2:]:
        #         #         try:
        #         #                 v = i
        #         #         except IndexError:
        #         #                 pass

        #         pool = np.array(self.data).flatten()
        #         # print(f"'pool':{pool}")
        #         data = shuffle(pool)
        #         # print(f'Shuffled Data: {data}')
                
        #         v1 = self.resample(data, size=len(v1), replace=False)
        #         v2 = self.resample(data, size=len(v2), replace=False)
        #         return v1, v2

        # def resample(self, x, size, replace = False):
        #          return np.random.choice(x, size=size, replace=replace)


        # def pvalue(self, iter = 1000):
        #         self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]
        #         # print(f' Observed Difference: {self.actual}')

        #         count = sum(1 for i in self.permute_dist if i >= self.actual)
        #         return count/iter




# class TwoGroups(PermutationTest):

#         def test_stat(self, data):
#                 v1, v2 = data
#                 return abs(v1.mean() - v2.mean())
#                 # return abs(v1.mean() - v2.mean())









# ##############################
# ### DESKRIPTIV STATISZTIKA ###
# ##############################

# def elemszam(adat):
#     print(adat.shape[0])
#     return adat.shape[0]



# # Check for a given parameter in jellemzok.json
# def check(col, mit_akarok_csekkolni, hol):
#     with open('minta.json', 'r') as f:
#         json_object_data = json.load(f)
#         if mit_akarok_csekkolni in json_object_data[col]:
#             return True



# # Check for a given parameter in jellemzok.json
# def check(col, mit_akarok_csekkolni, hol):
#     with open('minta.json', 'r') as f:
#         json_object_data = json.load(f)
#         if mit_akarok_csekkolni in json_object_data[col]:
#             return True



# # Descriptive Statistics

# ## Interval-ratio scale
# def descriptive_interval(col):
#     #TODO: az age kategóriát automatikusan bin-ekre osztani
#     atlag = np.round(col.mean())
#     print(atlag)
#     variancia = np.round(col.var())
#     szoras = np.round(col.std())

#     iqr_25 = np.round(col.quantile(0.25))
#     iqr_75 = np.round(col.quantile(0.75))
#     iqr = iqr_75 - iqr_25

#     maximum = np.round(col.max())
#     minimum = np.round(col.min())

#     leiro_stat = {
#         'atlag': atlag,
#         'variancia': variancia,
#         'szoras': szoras,
#         'iqr': iqr,
#         'max': maximum,
#         'min': minimum
#     }
#     return leiro_stat

# ## Ordinal scale

# def descriptive_ordinal(col):
#     print('ordinal func')
#     ordinal_dict = {}
#     kategoriak_szama = len(col.unique())
#     for kategoria in range(kategoriak_szama):
#         print(kategoria)
#         kategoria_neve = col.unique()[kategoria] # or col.value_counts().index[kategoria]
#         darabszam = col.value_counts()[kategoria]
#         iras(f'A vizsgálati személyek közül a {col.name} csoport tekintetében {darabszam} fő tartozott a {kategoria_neve} csoportba ')

#         ordinal_dict[kategoria_neve] = darabszam
#     print(ordinal_dict)
#     return ordinal_dict

# ## Nominal scale
# def descriptive_nominal(col):
#     print('nominal func \n')
#     ordinal_dict = {}
#     kategoriak_szama = len(col.unique())
#     for kategoria in range(kategoriak_szama):
#         print(kategoria)
#         kategoria_neve = col.unique()[kategoria] # or col.value_counts().index[kategoria]
#         darabszam = col.value_counts()[kategoria]
#         iras(f'A vizsgálati személyek közül a {col.name} csoport tekintetében {darabszam} fő tartozott a {kategoria_neve} csoportba ')

#         ordinal_dict[kategoria_neve] = darabszam
#     print(ordinal_dict)
#     return ordinal_dict    



# #################
# #################
# #################



# def mean(scores): # first moment
#     return sum(scores)/len(scores)

# def deviation(scores):
#     # Átlagos eltérés: az egyes adatok számtani átlagtól való abszolút eltéréseinek átlaga.
#     return np.round(sum(abs(x - mean(scores)) for x in scores) / (len(scores)-1))

# def covariance(x,y):
#     # Deviations from mean
#     d_x = x - x.mean()
#     d_y = y - y.mean()

#     cov = np.dot(d_x,d_y)/x.shape[0] 
#     return cov


# def std(scores): 
#     return 

# def var(scores): # second moment
#     pass


# def percentilerank(scores, your_score):
#     count = 0
#     for score in scores:
#         if score <= your_score:
#             count += 1
#     percentile_rank = 100.0 * count / len(scores)
#     return percentile_rank


# def z_score(scores):
#     return (scores - scores.mean())/scores.std()


# def factorial(x):
#     if x == 1:
#         return 1
#     else:
#         return x * factorial(x-1)


# def permutation(x):
#     pass



# class EffectSize:
#     def __init__(self):
#         pass


#     def prop(self, table: [list, np.ndarray, pd.core.series.Series], method='arr') -> int:
#         """
#         Risk of a particular event (e.g heart attack) happening.

#         Input 
#             2x2 matrix/table (pandas, numpy)
#             # List: [[90,10],[79,21]] (Not implemented)

#                 | Positive  | Negative
#         Group A |     90    |  10
#         Group B |     79    |  21

#         Returns
#             One of the called method results.
#         """

#         def arr():
#             """
#             Absolute Risk Reduction 
    
#             C - the proportion of people in the control group with the ailment of interest (illness)  
#             T - the proportion in the treatment group.

#             Then, ARR = C-T
#             """

#             C = table.iloc[0,1]/table.iloc[0].sum() # Negative Outcome / Sum (Group 1)
#             T = table.iloc[1,1]/table.iloc[1].sum()  # Negative Outcome / Sum (Group 2)
#             return np.round(abs(C-T), 3)

#         def rrr():
#             """
#             Relative Risk Reduction (RRR)
            
#             Measure the difference in terms of percentages.


#             """
#             return (C-T)/C * 100

#         def odds_ratio():
#             pass


#         def fourth_measure():
#             """
#             4th Measure:
#             The number of people who need to be treated in order to
#             prevent one person from having the ailment of interest.
#             """
#             return 1/arr()

#         if method == 'arr':
#             return arr()

#         if method == 'rrr':
#             return rrr()

#         if method == 'odds_ratio':
#             return odds_ratio()

#         if method == 'fourth_measure':
#             return fourth_measure()

#     def diff_means(self):
#         pass
 



# ##############################
# ### ELOSZLASOK ##############
# ##############################








# ##############################
# ### ASSUMPTIONS ##############
# ##############################


# # Distribution
# def distribution(col):
#     skewness = col.skew()
#     kurtosis = col.kurtosis()
#     return skewness, kurtosis

# #!/usr/bin/env python

# import scipy.stats as ss
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import logging
# import unittest
# import pingouin as pg

# # TODO
# # error handling
# # variable types
# # unittests

# def variable_type(x):
#     pass


# def simple_hist(x):
#     # legyen egy histogram es a normal eloszlas pdf-je
#     x = pd.DataFrame(x).hist()
#     plt.savefig('normality_test'.png)



# def box_whiskers():
#     pass


# def kurtosis(x):
#     return pd.Series(x).kurtosis()


# def skew(x):
#     return pd.Series(x).skew()

# def normality_test(x, method='shapiro-wilk'):
#     """

#     Returns tuple (test statistic, p-value)
#     """
#     df = len(x) - 1

#     krts = kurtosis(x)
#     skw = skew(x)


#     if method == 'shapiro-wilk':
#         w, p = np.round(ss.shapiro(x),3)  # test stat, p-value
#         print('ez jo')
#         if p >= 0.05:
#             print(f"The normality of {x.name} (majd oszlopnevet behelyettesiteni) scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.")
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
#         else:
#             print(f'The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. ")       

#         return (w, p)

#     elif method == 'kolmogorov-szmirnov':
#         k, p = np.round(ss.kstest(x, 'norm'),3)
#         if p >= 0.05:
#             print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were normally distributed (W({df}))={k}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
#         else:
#             print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were not normally distributed (W({df}))={k}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder. ")
#         return (k, p)
#     else:
#         raise ValueError('Only Shapiro-Wilk and Kolmogorov-Szmirnov are optional')


# def homogeneity_of_variance(*data):
#     """

#     Returns tuple (test statistic, p-value)
#     """
#     f, p = np.round(ss.levene(*data), 3)
#     df_b = 0
#     df_w = 0
#     # Pingouin módszere, df, scores, groups
#     #pg.homoscedasticity(data=df_anova, dv=scores, group=groups)

#     if p >= 0.05:
#         print(f"Levene's test for equality of variances is not significant (F({df_b},{df_w})= {f}, p={p}).")
#     else:
#         print(f' The homogeneity of variance assumption was violated (F({df_b},{df_w})= {f}, p={p}).')

#     return f, p




# def normality_test_decorator(x, method='shapiro-wilk'):
#     """
#     Maint to by executed as a decorator
#     Executed a separate function returns tuple (test statistic, p-value)

#     """
#     df = len(x) - 1

#     krts = kurtosis(x)
#     skw = skew(x)


#     if method == 'shapiro-wilk':
#         w, p = np.round(ss.shapiro(x),3)  # test stat, p-value
#         print('ez jo')
#         if p >= 0.05:
#             print(f"The normality of {x.name} (majd oszlopnevet behelyettesiteni) scores was assessed. The Shapiro-Wilk tests indicated that the scores were normally distributed (W({df}))={w}, p={p}.")
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
#         else:
#             print(f'The normality of {x.name} scores was assessed. The Shapiro-Wilk tests indicated that the scores were not normally distributed (W({df}))={w}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. ")
        
#         assmp = dict()
#         assmp['Shapiro-Wilk Test Statistic'] = w
#         assmp['p-value'] = p         

#         return (w, p)

#     elif method == 'kolmogorov-szmirnov':
#         k, p = np.round(ss.kstest(x, 'norm'),3)
#         if p >= 0.05:
#             print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were normally distributed (W({df}))={k}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder.")
#         else:
#             print(f'The normality of {x.name} scores was assessed. The Kolmogorov-Szmirnov tests indicated that the scores were not normally distributed (W({df}))={k}, p={p}.')
#             print(f"The variable's skew is {skw} and kurtosis is {krts}. You can find the histogram of the variable in this folder. ")
#         return (k, p)
#     else:
#         raise ValueError('Only Shapiro-Wilk and Kolmogorov-Szmirnov are optional')




# if __name__ == '__main__':
#     import unittest

#     np.random.seed(42)
#     x = pd.Series(np.random.normal(loc=6, scale=5, size=30), name='oszlop_nev1')
#     y = pd.Series(np.random.normal(loc=10, scale=3, size=30), name='oszlop_nev2')
#     z = pd.Series(np.random.normal(loc=5, scale=4, size=30), name='oszlop_nev3')

#     print(normality_test(x))
#     print(normality_test(x, method="kolmogorov-szmirnov"))

#     class TestAssumptions(unittest.TestCase):
#         def test_normality_test(self):
#             actual = normality_test(x, method='shapiro-wilk')
#             expected = (0.9751381874084473, 0.6868011355400085)
#             self.assertEqual(actual, expected)

#         def test_homogeneity_of_variance_1(self):
#             actual = homogeneity_of_variance(x, y, z)
#             self.assertEqual(
#                 actual, (1.914, 0.154), "OK with homogeneity of variance test!")

#         def test_homogeneity_of_variance_2(self):
#             actual = homogeneity_of_variance(x, y)
#             self.assertEqual(
#                 actual, (3.952, 0.052), "OK homogeneity of variance test!")






# ##############################
# ### KÉT CSOPORT ##############
# ##############################



# ### Kettő vagy több csoportra ellenőrzése
# def ketto_vagy_tobb_csoport_ellenorzese(df, groups):
#     """
#     ha kettőnél több csoport van -> return true
#     kettő csoport van ->return False
#     """
#     return df[groups].nunique() > 2




# ##############################
# ### MULTIPLE GROUPS ##########
# ##############################




# def anova_one_way():
#     pass

# def welch_f_test():
#     pass


# def kruskal_wallis():
#     pass





# ##############################
# ### EFFECT SIZE ##########
# ##############################












# ##############################
# ### CONFIDENCE INTERVAL ######
# ##############################


# #!/usr/bin/env python

# from scipy import stats
# import scipy
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.utils import shuffle


# #from descriptive import covariance


# class PermutationTest:

#     def __init__(self, *data):
#         self.data = data        
#         self.actual = self.test_stat(data)

#     def permutation(self):
#         data = [np.array(datum, dtype=float) for datum in self.data]
#         elemszamok = [data[i].shape[0] for i in range(len(data))] # returns list
#         csoportszam = len(elemszamok)

#         pool = np.concatenate(data, axis=0)
#         pool = shuffle(pool) # shuffle pool in-place

#         # need list of arrays
#         data = [self.resample(pool, size=elemszamok[i]) for i in range(csoportszam)]
#         return data

#     def resample(self, x, size, replace=False):
#         return np.random.choice(x, size=size, replace=replace)

#     def pvalue(self, iter=1000, ci=True, ci_level=95):
#         # Permuted sampling distribution
#         self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]

#         # P-value
#         count = sum(1 for i in self.permute_dist if i >= self.actual)
#         print(count)

#         #TODO compute confidence interval using your own percentile function
#         # Bootstraped [bs] Confidence Interval
#         if ci:
#             statistics = sorted(self.permute_dist)
#             #print(statistics)
#             # Trim endpoints of resampled CI
#             trim = ((1 - (ci_level/100))/2)
#             endpoints = int(trim*1000)
#             trimmed_ci = statistics[endpoints:-endpoints]
#             lower, upper = min(trimmed_ci), max(trimmed_ci)


#         return np.round(count/iter, 3), lower, upper


# class DiffTwoMeans(PermutationTest):
#     """
#     Significance test for the difference between two groups.
#     Makes no distributional assertion.
#     """
#     def test_stat(self, data):
#         if len(data) > 2:
#             raise TypeError(f'In case of more than two groups, test with ANOVA (see there).')

#         v1, v2 = data
#         return abs(v1.mean() - v2.mean())
#         # return abs(v1.mean() - v2.mean())



# #TODO: fix rounding, create more reusable code, more tests


# class ConfidenceInterval:
#     def __init__(self, *data, alpha=0.05, ci_level=.95):
#         self.data = data
#         self.alpha = alpha
#         self.ci_level = ci_level


#     def forMean(self, variance='unknown', **kwargs):
#         """
#         Confidence Interval for the Mean
#         Defaults to unknown variance, use t-distribution
#         """
#         x = np.array(self.data)
#         #print(x.size)
#         #print(x)
#         if variance == 'unknown':
#             sem = x.std()/ np.sqrt(x.size)
#             df = x.size - 1
#             t_cl = stats.t.interval(self.alpha,df)[1]
#             #print(sem,df,t_cl)
#             lower_limit = x.mean() - t_cl * sem
#             upper_limit = x.mean() + t_cl * sem
#             #print(np.round(lower_limit,3), np.round(upper_limit,3))
#             return (np.round(lower_limit,3), np.round(upper_limit,3))


#         if variance == 'known':
#             var = kwargs['var']
#             sem = var / np.sqrt(x.shape[0])
#             z_cl = scipy.stats.norm.ppf(ci_level)
#             lower_limit = x.mean() - z_cl * sem
#             upper_limit = x.mean() + z_cl * sem
#             #print(lower, upper)
#             return (upper_limit, lower_limit)


            
#     def forDifferenceBetweenmeans(self, equal_n = True):
#         x,y = self.data
#         x_n, y_n = x.shape[0], y.shape[0]
#         #print(x,y)
#         if equal_n:
#             # Standard error (equal sample size)
#             mean_diff = x.mean(), y.mean()
#             mse = (x.var()**2 + y.var()**2)/2

#             SE = np.sqrt(2*mse/x_n)
#             t_cl = scipy.stats.t.interval(alpha,df)
#             df = x_n - 1 + y_n - 1


#             # Compute CI
#             lower = mean_diff - t_cl * SE
#             upper = mean_diff + t_cl * SE
#             return (lower, upper)
#         else:

#             SSE = sum((i - x.mean())**2 for i in x) + sum((i - y.mean())**2 for i in y)
#             df = x_n - 1 + y_n - 1

#             MSE = SSE/df

#             n_h = 2/ (1/x_n + 1/y_n) # harmonic mean

#             mean_diff = x.mean(), y.mean()

#             SE = np.sqrt(2*mse/n_h)

#             t_cl = scipy.stats.t.interval(alpha,df)

#             # Compute CI
#             lower = mean_diff - t_cl * SE
#             upper = mean_diff + t_cl * SE
#             #print(np.round(lower,3), np.round(upper,3))
#             return (np.round(lower,3), np.round(upper,3))


#     def forCorr(self):
#         """
#         1. Convert r to z'
#         2. Compute a confidence interval in terms of z'
#         3. Convert the confidence interval back to r.
#         """

#         x,y = self.data

#         r = np.corrcoef(x,y)

#         # Step 1
#         r_z = np.arctanh(r)
#         r_z = 0.5 * np.log(1+r/1-r)

#         # Step 2
#         SE = 1/np.sqrt(x.shape[0]-3)

#         z_ci = stats.norm.ppf(1-self.alpha/2)

#         # Compute CI
#         lower = r_z - z_ci * SE
#         upper = r_z + z_ci * SE

#         # Step 3
#         lower, upper = np.tanh((lower, upper))
#         return (lower, upper)



#     def forProp(self,favorable_outcome):
#         print(self.data)
#         x = np.array(self.data)
#         print(x)
#         p = pd.value_counts(x[0], normalize=True)[favorable_outcome]
#         n = x[0].shape[0]
#         print(f'Shape {n}')
#         SE_p = np.sqrt(p*(1-p)/n)

#         z_ci = stats.norm.ppf(1-self.alpha/2)

#         # Compute CI (+ correction for estimating discrete distribution)
#         lower = p - z_ci * SE_p - (0.5/n) 
#         upper = p + z_ci * SE_p + (0.5/n)
#         return (lower, upper)



# import unittest


# class TestSum(unittest.TestCase):

#     #z_prop = np.random.randint(0, 2, size=30)
#     #print(f'Prop values {z_prop}')


#     #ci_class = ConfidenceInterval(x,y)

#     def test_formean(self):
#         np.random.seed(42)
#         x = np.random.randint(5, 15, size=30)
#         y = np.random.randint(10, 15, size=30)
#         self.assertEqual(ConfidenceInterval(x).forMean(), (9.703, 9.763), "Should be (9.703, 9.763)")

#     def test_forDifferenceBetweenmeans(self):
#         np.random.seed(42)
#         x = np.random.randint(5, 15, size=30)
#         y = np.random.randint(10, 15, size=30)
#         self.assertEqual(ConfidenceInterval(x,y).forMean(), (10.864, 10.903), "Should be (10.363, 10.403)")

#     def test_forCorr(self):
#         pass


#     def test_forProp(self):
#         np.random.seed(42)
#         z_prop = np.random.randint(0, 2, size=30)
#         self.assertEqual(ConfidenceInterval(z_prop).forProp(favorable_outcome=0), (0.2714786270876806, 0.6618547062456528), "Soulhd be this")




# if __name__ == '__main__':
#     pass

# # create workable code from CLI






# ##############################
# ### POWER ####################
# ##############################



# import numpy as np
# import pandas as pd
# import sys
# import unittest
# from sklearn.utils import shuffle


# if len(sys.argv) > 1:
#     if sys.argv[1] == 'statistical_rigor':
#         power = sys.argv[1]
#         alpha = sys.argv[2]
#         effect_size = sys.argv[3]
#         sample_size = sys.argv[4]
#     else:
#         power = sys.argv[1]  # 0.8
#         alpha = sys.argv[2]  # 0.05
#         mean_x = sys.argv[3]
#         mean_y = sys.argv[4]
#         sample_size = sys.argv[5]


# class PermutationTest:

#     def __init__(self, *data):
#         self.data = data
#         self.actual = self.test_stat(data)

#     def permutation(self):
#         data = [np.array(datum, dtype=float) for datum in self.data]
#         elemszamok = [data[i].shape[0]
#                       for i in range(len(data))]  # returns list
#         csoportszam = len(elemszamok)

#         pool = np.concatenate(data, axis=0)
#         # print(f"'pool':{pool}")
#         pool = shuffle(pool)  # shuffle pool in-place
#         #print(f'Shuffled Data: {data}')

#         # need list of arrays
#         data = [self.resample(pool, size=elemszamok[i])
#                 for i in range(csoportszam)]
#         return data

#     def resample(self, x, size, replace=False):
#         return np.random.choice(x, size=size, replace=replace)

#     def pvalue(self, iter=1000):
#         self.permute_dist = [self.test_stat(
#             self.permutation()) for x in range(iter)]
#         #print(self.permute_dist)
#         count = sum(1 for i in self.permute_dist if i >= self.actual)
#         return np.round(count/iter, 3)


# class Power(PermutationTest):
#     """
#     Documentation
#     """

#     def test_stat(self, data, method="mean_difference"):
#         if method == 'mean_difference':
#             if len(data) > 2:
#                 raise TypeError("More than two groups")

#             x, y = data
#             return abs(x.mean() - y.mean())

#         if method == 'cohen':
#             pass

#         if method == 'need more tests':
#             pass

#     def estimate_power(self, num_runs=101):
#         x, y = self.data
#         power_count = 0

#         for i in range(num_runs):
#             resample_x = np.random.choice(x, len(x), replace=True)
#             resample_y = np.random.choice(y, len(y), replace=True)

#             p = self.pvalue()
#             if p < 0.05:
#                 power_count += 1
#         print('hello')
#         return power_count/num_runs


# if __name__ == '__main__':
#     x = np.random.randint(10, 15, size=30)
#     y = np.random.randint(16, 30, size=30)

#     p = Power(x,y)
#     print(p.pvalue())
#     print(p.estimate_power())
    




# ##############################
# ### PARAMETRIC ###############
# ##############################





# #!/usr/bin/env python


# # Importing libraries
# import numpy as np
# import pandas as pd
# from scipy.stats import shapiro, kstest
# # import seaborn as sns
# # sns.set_theme(style="whitegrid")

# # from viz import box_plot # sajat module







# import numpy as np
# import pandas as pd
# from scipy.stats import chi2


# class Chi_square:
# 	def __init__(self):
# 		pass


# 		#	data = self.data.groupby(col1)[col2].count()


# 	def teststatistic(self, data):

# 		# observed frequency
# 		observed = data

# 		k = len(observed) # number of category
# 		dof = k-1 # degrees of freedom

# 		# expected frequency
# 		expected_value = np.sum(observed)/k # expected_value
# 		expected = np.ones(k) * expected_value # expected frequency
# 		# expected = np.full(shape=k, fill_value=expected_value) 

# 		test_stat = np.sum((expected-observed)**2/expected)
# 		return test_stat, dof

# 	def one_way_tables(self, data: np.ndarray):
# 		"""Tests deviations of differences between 
# 		theoretically expected and observed frequencies"""
# 		test_stat, dof = self.teststatistic(data)

# 		p_value = 1 - chi2.cdf(test_stat, dof)
# 		return test_stat, p_value

# 	def contingency_tables(self, data):
# 		"""Tests the relationship between categorical variables"""
# 		# data should be lists of lists, array
# 		observed.loc['Row Total'] = observed.sum(axis=0)
# 		observed['Column Total'] = observed.sum(axis=1)
		
# 		total = observed.loc['Row Total', 'Column Total']
# 		exp_freq = observed[:]

# 		for i in exp_freq.index[:-1]:
# 		    for j in exp_freq.columns[:-1]:
# 		    	row_total = exp_freq.loc['Row Total', j]
# 		    	col_total = exp_freq.loc[i, 'Column Total']
# 		    	exp_val = (row_total * col_total)/total
# 		    	exp_freq.loc[i,j] = exp_val.round(3) 

# 		chi_square_stat = np.sum(np.sum((exp_freq-observed)**2/exp_freq)) # chi-square stat
# 		dof = (r-1)*(c-1)

# 		#p_value = 
# 		return dof, chi_square_stat, p_value

# 	def fisher_exact(self):
# 		pass


# if __name__ == '__main__':
# 	chi = Chi_square()
# 	x = chi.one_way_tables([8,9,19,5,8,11])
# 	print(x)




# # expected value

# for i in df.index[:-1]:
#     for j in df.columns[:-1]:
#         # print(df.loc[i,j])
#         row_total = df.loc['Row Total', j]
#         col_total = df.loc[i, 'Column Total']
#         exp_val = (row_total * col_total)/total
#         # print(ev)
#         df.loc[i,j] = exp_val.round(3)



# #!/usr/bin/env python

# from scipy import stats
# import numpy as np
# import pandas as pd
# from sklearn.utils import shuffle

# # My modules
# from hypothesis import PermutationTest


 
# class Correlation:
#     """
#     Computing the correlation of two (X,Y) variables. Hypothesis testing for correlation.
#     Available methods: Covariance, Pearson correlation, p-value and degrees of freedom.
#     """

#     def __init__(self, measurement_level='continuous',
#                     linearity=True, 
#                     normal_dist=True, 
#                     outliers=False):
#         self.measurement_level = measurement_level
#         self.linearity = linearity
#         self.normal_dist = normal_dist
#         self.outliers = outliers

#     def assumptions(self, x,y):
#         """Assumptions"""
        
#         x = np.asarray(x)
#         y = np.asarray(y)
        
#         assert x.shape[0] == y.shape[0]  # Related pairs
#         if self.measurement_levelurement != 'continuous':         # Continuous variables
#             raise Exception("Both level of measurement must be continuous.") # raise Exception("Message")


#     def covariance(self, x,y):
#         # Deviations from mean
#         d_x = x - x.mean()
#         d_y = y - y.mean()

#         cov = np.dot(d_x,d_y)/x.shape[0] 
#         return cov

#     def pearson(self, x,y, significance=True):
#         """Compute the Pearson correlation coefficient for two variables"""
#         r = self.covariance(x,y)/(x.std() * y.std())

#         # Compute p-value for correlation.
#         def pvalue(x,y):
#             t = r*np.sqrt(len(x) + len(y) - 2)/np.sqrt(1 - r**2) # t-statistic
#             dof = len(x) - 1 + len(y) - 1 # degrees of freedom
#             pvalue = stats.t.sf(t, dof)*2
#             return pvalue, dof

#         if significance:
#             p, dof = pvalue(x,y)
#             return r, p, dof
#         else:
#             return r
        

#     def spearman(self, x,y, significance=True):
#         x_rank = pd.Series(x).rank()
#         y_rank = pd.Series(y).rank()

#         rho = self.pearson(x_rank, y_rank, significance=False)

#         def pvalue(iters=1000):
#             resample = []
#             for iter in range(iters):
#                 #x_rank_shuffled = shuffle(x_rank)
#                 y_rank_shuffled = shuffle(y_rank)
#                 resample.append(self.pearson(x_rank, y_rank_shuffled, significance=False))

#             count = sum(1 for i in resample if i >= rho)
#             return count/iters

#         if significance:
#             p = pvalue()
#             return rho, p
#         else:
#             return rho


    
# if __name__ == '__main__':
#     np.random.seed(42)
#     x = np.random.randint(10,15, size=30)
#     y = np.random.randint(10,15, size=30)

#     c =  Correlation()

#     r, pvalue, dof  = c.pearson(x,y)
#     print(r, pvalue, dof)
#     s_r, s_p = stats.pearsonr(x,y)
#     print(f'Scipy Pearson r {s_r} and p {s_p}')
    
    
#     print('\n')
    
    
#     r, pvalue  = c.spearman(x,y)
#     print(f'rho: {r}, p érték: {pvalue}')
#     precise_rho = pd.Series(x).corr(pd.Series(y), method='spearman')

#     print(f'igazi rho{precise_rho}')



# # CI


# #!/usr/bin/env python

# from scipy.stats import t
# import numpy as np


# # FIX
# # Unequal sample size automate & calculate sample size
# # CI for correlation, proportion
# # Integrate 2 function

# def interval(*variables, ci=0.95, compute = 'mean', sample_size='equal', pop_var = 'unknown'):

#     n1 = variables[0]
#     try:
#         n2 = variables[1]
#     except (ValueError, IndexError):
#         pass

#     # If the pop standard deviation is known
#     if pop_var == 'known':
#         n = len(n1)
#         sem = std/np.sqrt(n) 
#         if ci == 0.95:
#             lower = x.mean() - 1.96* sem
#             upper = x.mean() + 1.96* sem
#             return tuple(lower, upper)
#         if ci == 0.90: # or 99
#             pass
#     # If the pop standard deviation is unknown
#     else:    
#         if compute == 'mean':
#             n = len(n1)
#             m = n1.mean()

#             sem = (n1.std()/np.sqrt(n))
#             dof = n-1

#             t_cl = t.interval(ci, dof) # return an interval
#             t_cl = abs(t_cl[1]) # need one value
#             print(f't_cl is {t_cl}; sem is {sem}')

#             lower = m - t_cl * sem
#             upper = m + t_cl * sem
#             return tuple((lower, upper))

#         if compute == 'difference between means':

#             n = len(n1)
#             # Compute means
#             m1 = n1.mean()
#             m2 = n2.mean()

#             # Compute standard error of the difference between means
#             MSE = n1.std()- n2.std()/2
#             sem_m1_m2 = np.sqrt((2*MSE)/n)

#             # Degrees of freedom
#             dof = len(n1)-1 + len(n2) - 1

#             # t-distribution
#             t_cl = t.interval(ci, dof) # return an interval
#             t_cl = abs(t_cl[1]) # need one value
#             # print(f't_cl is {t_cl}; sem is {sem}')

#             m_diff = m1-m2
#             lower = m_diff - t_cl * sem_m1_m2
#             upper = m_diff + t_cl * sem_m1_m2
#             return tuple((lower, upper))
#         if compute == 'corr':
#             pass
#         if compute == 'proportion':
#             pass
    



# def interval(*variables, ci=95, method='bootstrap', iters=1000):
#     x,y = variables
#     means = sorted(list((np.mean(np.random.choice(x, size=len(x), replace=True)) for i in range(iters))))
#     # for i in range(iters):
#     #     x = np.random.choice(x, size=len(x), replace=True)
#     #     means.append(x.mean())
    

#     # Trim endpoints of resampled CI
#     trim = ((1 - (ci/100))/2)
#     endpoints = int(trim*1000)
#     trimmed_ci = means[endpoints:-endpoints]
#     lower, upper = min(trimmed_ci), max(trimmed_ci)

#     print(trim, endpoints, trimmed_ci, lower, upper)

#     # or 
#     trimmed_ci = means[25:]

#     return lower, upper



# if __name__ == '__main__':
#     np.random.seed(42)
#     n1 = np.random.randint(3,10, size=25)
#     n2 = np.random.randint(5,10, size=10)


#     lower, upper = interval(n1,n2, ci=95)
#     print(f'CI Lower {lower}, Upper {upper}, Mean: {n1.mean()}')





# ### My t_test package in production
# import numpy as np
# import pandas as pd
# from scipy import stats
# from scipy.stats import t



# from correlation import Correlation


# class LinearRegression:
# 	'''
# 	Simple Linear Regression

# 	X: Feature, Predictor, Independent variable
# 	Y: Criterion, Dependent variable

# 	Formula:
# 		Y_hat = intercept + slope*X
# 	'''

# 	def __init__(self, homoscedasticity=True):
# 		self.homoscedasticity = homoscedasticity # Assumptions 1
# 		self.corr = Correlation


# 		# Assumptions
# 		# linear relationship
# 		# normality for the residuals
	
# 	def linear_regression(self, X, Y, normality=True, standardize=False):

# 		if not isinstance(X, np.ndarray):
# 			X = np.asarray(X)
# 		if not isinstance(Y, np.ndarray):
# 			Y = np.asarray(Y)


# 		# Sample size, degrees of freedom
# 		N = x.size + y.size
# 		df = N-2

# 		# Intercept, slope

# 		# Method 1: Linear Algebra
# 		X_b = np.c_[np.ones((X.size, 1)), X] # add x_0 = 1 to each instance
# 		intercept, coeff = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
# 		# return intercept, coeff
		
# 		self.coeff = coeff
# 		self.intercept = intercept

# 		print(intercept, coeff)

# 		# Method 2: Statistics
# 		r = self.corr().pearson(x,y, significance=False)

	
# 		if standardize:
# 			slope = r
# 			intercept = 0
# 			y_pred = np.array([(slope*X + intercept) for X in X])
# 		else:
# 			slope = r*(x.std()/y.std())
# 			intercept = x.mean()- slope*x.mean()

# 		print(intercept, slope)
# 		# return intercept, slope

# 		y_pred = np.array([slope*X + intercept for x in X])


# 		# Partitioning Sums of Squares
# 		# Buggy

# 		# SSY = sum([(i-y_pred.mean())**2 for i in y])
# 		# SSY_pred = sum([(i-y_pred.mean())**2 for i in y_pred])
# 		# SSE = sum([(y_hat-i)**2 for y_hat,i in zip(y_pred, y)])


# 		# if SSY == SSY_pred + SSE:
# 		# 	print('Everythngs fine')

# 		# else: 
# 		# 	print("The SSY and the SSY' and SSE is not equal, do something..." )

# 		# proportion_explained = SSY_pred/SSY
# 		# proportion_not_explained = SSE/SSY


# 		# # summary_table = pd.DataFrame({'Source': ['Explained', 'Error', 'Total'], 'Sum of Squares': [SSY_predicted, SSE, SSY], 'df': [1, N-2, N], 'Mean Square': ['Nan', 'Nan','Nan']})
# 		# # print(summary_table)
		
# 		# s_est = np.sqrt(SSE/N-2)
# 		# s_est_corr = np.sqrt((1-r**2)*SSY/N-2)
# 		# if s_est_corr == s_est:
# 		# 	print('Everythngs fine')
# 		# else:
# 		# 	print('The standard error of the estimate is not right.')


# 		# SSX = sum([(i- x.mean())**2 for i in x])
# 		# standard_error_of_slope = s_est/SSX

# 		# t_statistic = slope/standard_error_of_slope

# 		# pvalue = stats.t.sf(t_statistic, df)*2 # two sided p-value

# 		# # Confidence interval
# 		# t_95 = t.interval(0.95, df) #t.95 for confidence interval

# 		# CI_upper = slope + t_95[1] * standard_error_of_slope
# 		# CI_lower = slope - t_95[1] * standard_error_of_slope
# 		# # print('The CI for the slope is: {CI_lower}{CI_upper}')

# 		# return t_statistic, pvalue, tuple((CI_lower, CI_upper))


# 		# # Influental values (Cook's distance)
# 		# # Beta weight
# 		# # R^2 - proportion of variance explained



# if __name__ == '__main__':
# 	np.random.seed(42)
# 	x = np.random.randint(10,15, size=30)
# 	y = np.random.randint(10,15, size=30)

# 	r = LinearRegression()

# 	r.linear_regression(x,y)

# 	print('\n')
# 	print(r.intercept, r.coeff)
# 	# print(f'P: {t_statistic}, P {pvalue}, CI {ci}')




# #!/usr/bin/env python3

# import numpy as np


# # Standard error of the mean
# def sem(x, *vars, method='one mean'):
# 	"""
# 	Compute the estimate of the standard error of the statistic in case of two groups. 
# 	"""


# 	if method == 'one mean':
# 		return x.var()/np.sqrt(len(x))

# 	if method == 'difference between means':
# 		y = vars[0]
# 		mse = (x.var() - y.var())/2 # n is equal the number of scores in each group
# 		return np.sqrt(2*mse/len(x))


# if __name__ == '__main__':
# 	np.random.seed(42)
# 	x = np.random.randint(10,15, size=30)
# 	y = np.random.randint(10,15, size=30)

# 	se = sem(x, method='one mean')
# 	print(se)

# 	se = sem(x,y, method = 'difference between means')
# 	print(se)



# #!/usr/bin/env python

# import numpy as np
# import pandas as pd
# import seaborn as sns 
# from scipy import stats
# from scipy.stats import t

# import standard_error

# from writer import Writer


# class t_test:
#     def __init__(self, variance_equal=True, normality=True, sample_size_equal = True, independent_sample=True,):
#         # Checking for assumptions across all t-tests
#         self.variance_equal = variance_equal # homogenity of variance
#         # MISSING normal distribution
#         self.independent_sample = independent_sample # each value is sampled indenpendently
#         self.sample_size_equal = sample_size_equal # equal size

#         self.sem = standard_error.Sem()
#         self.writer = Writer()



#     def t_statistic(self, statistic, sem,hypothesized_value=0):
#         """Compute t-statistic"""
#         return statistic - hypothesized_value/sem
        


#     def one_sample(self, n1, n2, hypothesized_value=0):
#         statistic = np.array(n1) - np.array(n2)
#         # sem = statistic.var()/np.sqrt(len(n1)) # standard error of the statistic
#         sem = self.sem.sem_mean(statistic, n1)

#         t_statistic = self.t_statistic(statistic.mean(),sem)
#         df = n1.shape[0] - 1 # degrees of freedom
        
#         p_value = stats.t.sf(t_statistic, df) # two sided p-value

#         h0_false = f"If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}. \
#         The null hypothesis that the population mean difference score is zero can be rejected. \
#         The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition."

#         h0_not_rejected = f"The null hypothesis - that the population mean difference score is zero - cannot be rejected."

#         if p_value < 0.05:
#             self.writer.generic_writer(h0_false)
#             print(f'If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}.')
#             print('The null hypothesis that the population mean difference score is zero can be rejected.')
#             print('The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition.') 
#         else:
#             self.writer.generic_writer(h0_not_rejected)
#             print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')


#     def indenpendent_sample(self, n1, n2, hypothesized_value=0):
#         """Test for difference between means from two separate groups of subjects."""

#         # Compute the statistic
#         statistic = n1.mean() - n2.mean()

#         # Compute the estimate of the standard error of the statistic
#         mse = (n1.var() - n2.var())/2
#         sem = np.sqrt(2*mse/n1.shape[0]) # n is equal the number of scores in each group
#         # sem = S_{m_1-m_2}

#         # sem = self.sem.sem_mean2(n1,n2)

#         # Compute t
#         t_statistic = statistic/sem
#         df = (n1.shape[0]) - 1 + (n2.shape[0] - 1)
#         # Compare areas of the t distribution
#         p_value = stats.t.sf(t_statistic, df)

#         # Interpretation
#         if p_value < 0.05:
#             print(f'If the treatment/experiment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low (t={t_statistic} p={p_value}).')
#             print('The null hypothesis that the population mean difference score is zero can be rejected.')
#             print('The conclusion is that the population mean for the treatment/experiment condition is higher than the population mean for the control condition.') 
#         else:
#             print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')





# def main():
#     n1 = np.array([10,12,13,14,15,13,14,13,12])    # n1 = 5 * np.random.randint(20, size=40)
#     n2 = np.array([1,2,3,4,5,3,4,3,2])     # n2 = 5 * np.random.randint(5, size=40)

#     t_teszt = t_test()
#     t_teszt.one_sample(n1,n2)


#     n1 = np.array(np.random.randint(low=5,high=15, size=30))    # n1 = 5 * np.random.randint(20, size=40)
#     n2 = np.array(np.random.randint(low=10, high=15, size=30))     # n2 = 5 * np.random.randint(5, size=40)

#     t_teszt = t_test()
#     t_teszt.indenpendent_sample(n1,n2)

# if __name__ == "__main__":
#     main() 











# ##############################
# ### NON-PARAMETRIC ###############
# ##############################


# #!/usr/bin/env python

# from scipy import stats
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.utils import shuffle



# from descriptive import covariance


# class PermutationTest:

#     def __init__(self, *data):
#         self.data = data
#         self.actual = self.test_stat(data)

#     def permutation(self):

        
#         data = [np.array(datum, dtype=float) for datum in self.data]
#         elemszamok = [data[i].shape[0] for i in range(len(data))] # returns list
#         csoportszam = len(elemszamok)


#         pool = np.concatenate(data, axis=0)
#         #print(f"'pool':{pool}")
#         pool = shuffle(pool) # shuffle pool in-place
#         #print(f'Shuffled Data: {data}')


#         # need list of arrays
#         data = [self.resample(pool, size=elemszamok[i]) for i in range(csoportszam)]
#         return data

#     def resample(self, x, size, replace=False):
#         return np.random.choice(x, size=size, replace=replace)

#     def pvalue(self, iter=1000):
#         self.permute_dist = [self.test_stat(self.permutation()) for x in range(iter)]
#         #print(self.permute_dist)
#         count = sum(1 for i in self.permute_dist if i >= self.actual)
#         return np.round(count/iter, 3)


# class DiffTwoMeans(PermutationTest):
#     """
#     Significance test for the difference between two groups.
#     Makes no distributional assertion.
#     """
#     def test_stat(self, data):
#         if len(data) > 2:
#             raise TypeError(f'In case of more than two groups, test with ANOVA (see there).')

#         v1, v2 = data
#         return abs(v1.mean() - v2.mean())
#         # return abs(v1.mean() - v2.mean())


# class PermuteCorr(PermutationTest):
#     '''Testing the significance of r that makes no distributional assumptions.'''

#     def test_stat(self, data):  # Association
#         x,y = data
#         test_stat = covariance(x,y)/(x.std() * y.std())
#         # test_stat = np.corrcoef(x,y)[0][1]
#         # test_stat = pd.Series(x).corr(pd.Series(y))
#         return test_stat





# class MultiGroup(PermutationTest):
#     """
#     ANOVA is used to test differences among means by analyzing variance.

#     Test uses resampling procedure (without making any distributional assumptions).

#     H_0: All X population means are equal (omnibus hypothesis).
#     H_1: At least one population mean is different from at least one other mean.
#     """
#     def test_stat(self, data):
#         if len(data) <= 2:
#             raise TypeError(f'In case of two groups, test with t-test (see there).')

#         #data = [np.array(datum, dtype=float) for datum in data]

#         # MSE
#         # Mean of sample variances
#         mse = np.mean([data[i].var() for i in range(len(data))])

#         # MSB
#         # Compute means
#         # Compute variance of means
#         # Multiply by n
#         msb = np.var([data[i].mean() for i in range(len(data))]) * data[0].shape[0]

#         # F-ratio
#         f = msb/mse
#         return f

#         # alldata = np.concatenate(data, axis=1)

#         # bign = alldata.shape[axis]

#         # print(alldata, bign)
#         # print('test')




# #############################


# class MannWhitneyU:
#     """
#     Rank randomization tests for differences in central tendency. Also called Wilcoxon rank-sum test
#     """

#     def __init__(self):
#         pass

#     def test_stat(self, *data, pvalue=True, alternative='two-sided', distribution='different'):
#         x, y = data

#         if not isinstance(x, pd.core.series.Series):
#             x = pd.Series(x)
#         if not isinstance(y, pd.core.series.Series):
#             y = pd.Series(y)

#         n1 = len(x)
#         n2 = len(y)
#         data = pd.concat((x, y)).rank()
#         x, y = data[:n1], data[n1:]

#         if distribution == 'different':

#             all_rank_sum = n1*(n1+1)/2  # all rank sum
#             R_1 = sum(x)  # sum rank of group 1
#             R_2 = sum(y)  # or all_rank_sum - R_x

#             U1 = R_1 - ((n1*(n1+1))/2)

#             U2 = R_2 - ((n2*(n2+1))/2)
#             U = min(U1, U2)

#             Umean = (n1*n2)/2
#             Ustd = np.sqrt(n1*n2*(n1+n2+1)/12)

#         elif distribution == 'identical':  # identical distribution
#             x, y = data
#             x_median = pd.Series(x).median()
#             y_median = pd.Series(y).median()
#             # return abs(x_median - y_median) # difference between sum of ranks; two-tailed
#         else:
#             raise Exception(
#                 'You should specify the distribution parameter. Available parameters: "identical", "different".')

#         def pvalue(alternative=alternative):
#             # For ldatae samples, U is approximately normally distributed. In that case, the standardized value equals to
#             z = (U - Umean) / Ustd
#             if alternative == 'two-sided':
#                 p = 2*stats.norm.sf(abs(z))
#             elif alternative == 'one-sided':
#                 p = 1 - stats.norm.cdf(abs(z))
#             else:
#                 raise Exception('Hypothesis test should be one or two sided.')
#             return p
#         if pvalue:
#             p = pvalue()
#         return tuple((U, p))
