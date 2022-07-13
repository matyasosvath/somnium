#!/usr/bin/env python


from typing import List
from correlation.icorrelation import ICorrelation
from variable import Variable
from result import Result
import numpy as np

import pingouin as pg

class AbstractCorrelation(ICorrelation):
    """
    Interface for correlation.
    """
    def __init__(self) -> None:
        super().__init__()

    def correlate(data1: Variable, data2: Variable) -> Result: 
        """
        (Robust) correlation between two variables, implementing the correct correlation type for data.
        """
        raise NotImplementedError()
    
    def phi_coefficient(self, data1: Variable, data2: Variable) -> Result:
        raise NotImplementedError()

    def matthews_coefficient(self, data1: Variable, data2: Variable) -> Result: 
        raise NotImplementedError()
    
    def rank_biserial_coefficient(self, data1: Variable, data2: Variable) -> Result: 
        raise NotImplementedError()
    
    def kendall_tau_b(self, data1: Variable, data2: Variable) -> Result:
        """
        Kendall's tau-b correlation (for ordinal data).
        """
        return pg.corr(data1.values, data2.values, method="kendall").round(3)

    #region Rank based correlations

    def spearman_rank_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(self, np.array(data1.values), np.array(data2.values), method="spearman").round(3)

    def shepherd_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(data1, data2, method="shepherd").round(3)
    
    def skipped_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(data1, data2, method="skipped").round(3)
    
    def percentage_bend_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(data1, data2, method="percbend").round(3)
    
    def biweight_midcorrelation_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(data1, data2, method="bicor").round(3)

    #endregion

    def pearson_coefficient(self, data1: Variable, data2: Variable) -> Result:
        return pg.corr(data1.values, data2.values, method="pearson").round(3)

    def pairwise_correlation_coefficient(self, data1: Variable, data2: Variable) -> Result: 
        raise NotImplementedError()

    




