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