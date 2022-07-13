#!/usr/bin/env python

from typing import List
from variable import Variable
from result import Result

class ICorrelation(object):
    """
    Interface for correlation.
    """   
    def phi_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def matthews_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def rank_biserial_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def kendall_tau_b(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def spearman_rank_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()

    def shepherd_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def skipped_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def percentage_bend_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()
    
    def biweight_midcorrelation_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()

    def pearson_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()

    def pairwise_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result: raise NotImplementedError()







