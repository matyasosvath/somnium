#!/usr/bin/env python

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