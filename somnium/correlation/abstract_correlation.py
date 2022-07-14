#!/usr/bin/env python


from typing import Iterable
from result import Result
from variable import Variable

from assumption.assumption import Assumption
from correlation.icorrelation import ICorrelation
from logger.ilogger import ILogger
from visualization.ivisualization import IVisualize
from writer.IWriter import IWriter

import pingouin as pg


class AbstractCorrelation(ICorrelation):
    def __init__(self, assumption: Assumption, visualize: IVisualize, writer: IWriter, logger: ILogger = None):
        self.__assumption = assumption
        self.__visualization = visualize
        self.__writer = writer
        self.logger = logger

    @property
    def assumption(self):
        return self.__assumption

    @property
    def visualization(self):
        return self.__visualization

    @property
    def writer(self):
        return self.__writer
    

    def correlate(data1: Variable, data2: Variable) -> Result: 
        """
        (Robust) correlation between two Iterable[float]s, 
        implementing the correct correlation type for data.
        """
        raise NotImplementedError()
    
    def phi_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        raise NotImplementedError()

    def matthews_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result: 
        raise NotImplementedError()
    
    def rank_biserial_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result: 
        raise NotImplementedError()
    
    def kendall_tau_b(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1.values, data2.values, method="kendall").round(3)

    #region Rank based correlations

    def spearman_rank_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1, data2, method="spearman").round(3)

    def shepherd_correlation_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1, data2, method="shepherd").round(3)
    
    def skipped_correlation_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1, data2, method="skipped").round(3)
    
    def percentage_bend_correlation_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1, data2, method="percbend").round(3)
    
    def biweight_midcorrelation_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1, data2, method="bicor").round(3)

    #endregion

    def pearson_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result:
        return pg.corr(data1.values, data2.values, method="pearson").round(3)

    def pairwise_correlation_coefficient(self, data1: Iterable[float], data2: Iterable[float]) -> Result: 
        raise NotImplementedError()

    




