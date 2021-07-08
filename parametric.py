#!/usr/bin/env python


# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import shapiro, kstest
# import seaborn as sns
# sns.set_theme(style="whitegrid")

# from viz import box_plot # sajat module







import numpy as np
import pandas as pd
from scipy.stats import chi2


class Chi_square:
	def __init__(self):
		pass


		#	data = self.data.groupby(col1)[col2].count()


	def teststatistic(self, data):

		# observed frequency
		observed = data

		k = len(observed) # number of category
		dof = k-1 # degrees of freedom

		# expected frequency
		expected_value = np.sum(observed)/k # expected_value
		expected = np.ones(k) * expected_value # expected frequency
		# expected = np.full(shape=k, fill_value=expected_value) 

		test_stat = np.sum((expected-observed)**2/expected)
		return test_stat, dof

	def one_way_tables(self, data: np.ndarray):
		"""Tests deviations of differences between 
		theoretically expected and observed frequencies"""
		test_stat, dof = self.teststatistic(data)

		p_value = 1 - chi2.cdf(test_stat, dof)
		return test_stat, p_value

	def contingency_tables(self, data):
		"""Tests the relationship between categorical variables"""
		# data should be lists of lists, array
		observed.loc['Row Total'] = observed.sum(axis=0)
		observed['Column Total'] = observed.sum(axis=1)
		
		total = observed.loc['Row Total', 'Column Total']
		exp_freq = observed[:]

		for i in exp_freq.index[:-1]:
		    for j in exp_freq.columns[:-1]:
		    	row_total = exp_freq.loc['Row Total', j]
		    	col_total = exp_freq.loc[i, 'Column Total']
		    	exp_val = (row_total * col_total)/total
		    	exp_freq.loc[i,j] = exp_val.round(3) 

		chi_square_stat = np.sum(np.sum((exp_freq-observed)**2/exp_freq)) # chi-square stat
		dof = (r-1)*(c-1)

		#p_value = 
		return dof, chi_square_stat, p_value

	def fisher_exact(self):
		pass


if __name__ == '__main__':
	chi = Chi_square()
	x = chi.one_way_tables([8,9,19,5,8,11])
	print(x)




# expected value

for i in df.index[:-1]:
    for j in df.columns[:-1]:
        # print(df.loc[i,j])
        row_total = df.loc['Row Total', j]
        col_total = df.loc[i, 'Column Total']
        exp_val = (row_total * col_total)/total
        # print(ev)
        df.loc[i,j] = exp_val.round(3)



#!/usr/bin/env python

from scipy import stats
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# My modules
from hypothesis import PermutationTest


 
class Correlation:
    """
    Computing the correlation of two (X,Y) variables. Hypothesis testing for correlation.
    Available methods: Covariance, Pearson correlation, p-value and degrees of freedom.
    """

    def __init__(self, measurement_level='continuous',
                    linearity=True, 
                    normal_dist=True, 
                    outliers=False):
        self.measurement_level = measurement_level
        self.linearity = linearity
        self.normal_dist = normal_dist
        self.outliers = outliers

    def assumptions(self, x,y):
        """Assumptions"""
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        assert x.shape[0] == y.shape[0]  # Related pairs
        if self.measurement_levelurement != 'continuous':         # Continuous variables
            raise Exception("Both level of measurement must be continuous.") # raise Exception("Message")


    def covariance(self, x,y):
        # Deviations from mean
        d_x = x - x.mean()
        d_y = y - y.mean()

        cov = np.dot(d_x,d_y)/x.shape[0] 
        return cov

    def pearson(self, x,y, significance=True):
        """Compute the Pearson correlation coefficient for two variables"""
        r = self.covariance(x,y)/(x.std() * y.std())

        # Compute p-value for correlation.
        def pvalue(x,y):
            t = r*np.sqrt(len(x) + len(y) - 2)/np.sqrt(1 - r**2) # t-statistic
            dof = len(x) - 1 + len(y) - 1 # degrees of freedom
            pvalue = stats.t.sf(t, dof)*2
            return pvalue, dof

        if significance:
            p, dof = pvalue(x,y)
            return r, p, dof
        else:
            return r
        

    def spearman(self, x,y, significance=True):
        x_rank = pd.Series(x).rank()
        y_rank = pd.Series(y).rank()

        rho = self.pearson(x_rank, y_rank, significance=False)

        def pvalue(iters=1000):
            resample = []
            for iter in range(iters):
                #x_rank_shuffled = shuffle(x_rank)
                y_rank_shuffled = shuffle(y_rank)
                resample.append(self.pearson(x_rank, y_rank_shuffled, significance=False))

            count = sum(1 for i in resample if i >= rho)
            return count/iters

        if significance:
            p = pvalue()
            return rho, p
        else:
            return rho


    
if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.randint(10,15, size=30)
    y = np.random.randint(10,15, size=30)

    c =  Correlation()

    r, pvalue, dof  = c.pearson(x,y)
    print(r, pvalue, dof)
    s_r, s_p = stats.pearsonr(x,y)
    print(f'Scipy Pearson r {s_r} and p {s_p}')
    
    
    print('\n')
    
    
    r, pvalue  = c.spearman(x,y)
    print(f'rho: {r}, p érték: {pvalue}')
    precise_rho = pd.Series(x).corr(pd.Series(y), method='spearman')

    print(f'igazi rho{precise_rho}')



# CI


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





### My t_test package in production
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t



from correlation import Correlation


class LinearRegression:
	'''
	Simple Linear Regression

	X: Feature, Predictor, Independent variable
	Y: Criterion, Dependent variable

	Formula:
		Y_hat = intercept + slope*X
	'''

	def __init__(self, homoscedasticity=True):
		self.homoscedasticity = homoscedasticity # Assumptions 1
		self.corr = Correlation


		# Assumptions
		# linear relationship
		# normality for the residuals
	
	def linear_regression(self, X, Y, normality=True, standardize=False):

		if not isinstance(X, np.ndarray):
			X = np.asarray(X)
		if not isinstance(Y, np.ndarray):
			Y = np.asarray(Y)


		# Sample size, degrees of freedom
		N = x.size + y.size
		df = N-2

		# Intercept, slope

		# Method 1: Linear Algebra
		X_b = np.c_[np.ones((X.size, 1)), X] # add x_0 = 1 to each instance
		intercept, coeff = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
		# return intercept, coeff
		
		self.coeff = coeff
		self.intercept = intercept

		print(intercept, coeff)

		# Method 2: Statistics
		r = self.corr().pearson(x,y, significance=False)

	
		if standardize:
			slope = r
			intercept = 0
			y_pred = np.array([(slope*X + intercept) for X in X])
		else:
			slope = r*(x.std()/y.std())
			intercept = x.mean()- slope*x.mean()

		print(intercept, slope)
		# return intercept, slope

		y_pred = np.array([slope*X + intercept for x in X])


		# Partitioning Sums of Squares
		# Buggy

		# SSY = sum([(i-y_pred.mean())**2 for i in y])
		# SSY_pred = sum([(i-y_pred.mean())**2 for i in y_pred])
		# SSE = sum([(y_hat-i)**2 for y_hat,i in zip(y_pred, y)])


		# if SSY == SSY_pred + SSE:
		# 	print('Everythngs fine')

		# else: 
		# 	print("The SSY and the SSY' and SSE is not equal, do something..." )

		# proportion_explained = SSY_pred/SSY
		# proportion_not_explained = SSE/SSY


		# # summary_table = pd.DataFrame({'Source': ['Explained', 'Error', 'Total'], 'Sum of Squares': [SSY_predicted, SSE, SSY], 'df': [1, N-2, N], 'Mean Square': ['Nan', 'Nan','Nan']})
		# # print(summary_table)
		
		# s_est = np.sqrt(SSE/N-2)
		# s_est_corr = np.sqrt((1-r**2)*SSY/N-2)
		# if s_est_corr == s_est:
		# 	print('Everythngs fine')
		# else:
		# 	print('The standard error of the estimate is not right.')


		# SSX = sum([(i- x.mean())**2 for i in x])
		# standard_error_of_slope = s_est/SSX

		# t_statistic = slope/standard_error_of_slope

		# pvalue = stats.t.sf(t_statistic, df)*2 # two sided p-value

		# # Confidence interval
		# t_95 = t.interval(0.95, df) #t.95 for confidence interval

		# CI_upper = slope + t_95[1] * standard_error_of_slope
		# CI_lower = slope - t_95[1] * standard_error_of_slope
		# # print('The CI for the slope is: {CI_lower}{CI_upper}')

		# return t_statistic, pvalue, tuple((CI_lower, CI_upper))


		# # Influental values (Cook's distance)
		# # Beta weight
		# # R^2 - proportion of variance explained



if __name__ == '__main__':
	np.random.seed(42)
	x = np.random.randint(10,15, size=30)
	y = np.random.randint(10,15, size=30)

	r = LinearRegression()

	r.linear_regression(x,y)

	print('\n')
	print(r.intercept, r.coeff)
	# print(f'P: {t_statistic}, P {pvalue}, CI {ci}')




#!/usr/bin/env python3

import numpy as np


# Standard error of the mean
def sem(x, *vars, method='one mean'):
	"""
	Compute the estimate of the standard error of the statistic in case of two groups. 
	"""


	if method == 'one mean':
		return x.var()/np.sqrt(len(x))

	if method == 'difference between means':
		y = vars[0]
		mse = (x.var() - y.var())/2 # n is equal the number of scores in each group
		return np.sqrt(2*mse/len(x))


if __name__ == '__main__':
	np.random.seed(42)
	x = np.random.randint(10,15, size=30)
	y = np.random.randint(10,15, size=30)

	se = sem(x, method='one mean')
	print(se)

	se = sem(x,y, method = 'difference between means')
	print(se)



#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns 
from scipy import stats
from scipy.stats import t

import standard_error

from writer import Writer


class t_test:
    def __init__(self, variance_equal=True, normality=True, sample_size_equal = True, independent_sample=True,):
        # Checking for assumptions across all t-tests
        self.variance_equal = variance_equal # homogenity of variance
        # MISSING normal distribution
        self.independent_sample = independent_sample # each value is sampled indenpendently
        self.sample_size_equal = sample_size_equal # equal size

        self.sem = standard_error.Sem()
        self.writer = Writer()



    def t_statistic(self, statistic, sem,hypothesized_value=0):
        """Compute t-statistic"""
        return statistic - hypothesized_value/sem
        


    def one_sample(self, n1, n2, hypothesized_value=0):
        statistic = np.array(n1) - np.array(n2)
        # sem = statistic.var()/np.sqrt(len(n1)) # standard error of the statistic
        sem = self.sem.sem_mean(statistic, n1)

        t_statistic = self.t_statistic(statistic.mean(),sem)
        df = n1.shape[0] - 1 # degrees of freedom
        
        p_value = stats.t.sf(t_statistic, df) # two sided p-value

        h0_false = f"If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}. \
        The null hypothesis that the population mean difference score is zero can be rejected. \
        The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition."

        h0_not_rejected = f"The null hypothesis - that the population mean difference score is zero - cannot be rejected."

        if p_value < 0.05:
            self.writer.generic_writer(h0_false)
            print(f'If the treatment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low t={t_statistic} p={p_value}.')
            print('The null hypothesis that the population mean difference score is zero can be rejected.')
            print('The conclusion is that the population mean for the treatment condition is higher than the population mean for the control condition.') 
        else:
            self.writer.generic_writer(h0_not_rejected)
            print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')


    def indenpendent_sample(self, n1, n2, hypothesized_value=0):
        """Test for difference between means from two separate groups of subjects."""

        # Compute the statistic
        statistic = n1.mean() - n2.mean()

        # Compute the estimate of the standard error of the statistic
        mse = (n1.var() - n2.var())/2
        sem = np.sqrt(2*mse/n1.shape[0]) # n is equal the number of scores in each group
        # sem = S_{m_1-m_2}

        # sem = self.sem.sem_mean2(n1,n2)

        # Compute t
        t_statistic = statistic/sem
        df = (n1.shape[0]) - 1 + (n2.shape[0] - 1)
        # Compare areas of the t distribution
        p_value = stats.t.sf(t_statistic, df)

        # Interpretation
        if p_value < 0.05:
            print(f'If the treatment/experiment had no effect, the probability of finding a difference between means as large or larger (in either direction) than the difference found is very low (t={t_statistic} p={p_value}).')
            print('The null hypothesis that the population mean difference score is zero can be rejected.')
            print('The conclusion is that the population mean for the treatment/experiment condition is higher than the population mean for the control condition.') 
        else:
            print('The null hypothesis - that the population mean difference score is zero - cannot be rejected.')





def main():
    n1 = np.array([10,12,13,14,15,13,14,13,12])    # n1 = 5 * np.random.randint(20, size=40)
    n2 = np.array([1,2,3,4,5,3,4,3,2])     # n2 = 5 * np.random.randint(5, size=40)

    t_teszt = t_test()
    t_teszt.one_sample(n1,n2)


    n1 = np.array(np.random.randint(low=5,high=15, size=30))    # n1 = 5 * np.random.randint(20, size=40)
    n2 = np.array(np.random.randint(low=10, high=15, size=30))     # n2 = 5 * np.random.randint(5, size=40)

    t_teszt = t_test()
    t_teszt.indenpendent_sample(n1,n2)

if __name__ == "__main__":
    main() 
