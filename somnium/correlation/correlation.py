#!/usr/bin/env python

from argparse import ArgumentError
from typing import List, Tuple
from collections.abc import Iterable
from hypothesistest import HypothesisTestPermute
from correlation.abstract_correlation import AbstractCorrelation
from assumption.assumption import Assumption

#from logging.console_logger  import ConsoleLogger
#from logging.ilogger import ILogger

from data import Data
from test_result import TestResult


class Correlation(HypothesisTestPermute, AbstractCorrelation):
    """
    
    """

    def __init__(self, assumption: Assumption, *data):
        
        super().__init__(data)

        self.assumption = assumption
        self.__assumption_result = dict()
        #self.logger = logger

    def correlate(self, remove_outliar=False) -> TestResult:
        """
        Full-process
        1. Remove outliers
        2. Normality test
        3. Data type check
        """
        if remove_outliar:
            self.data1 = self.assumption.remove_outliar(self.data1)
            self.data2 = self.assumption.remove_outliar(self.data2)

        self.__assumption_result["data1"] = self.assumption.check(self.data[0])
        self.__assumption_result["data2"] = self.assumption.check(self.data[1])

        #TODO visualize


        self.test_result = TestResult(
            len(self.pool), 
            self.actual, 
            self.p_value()
            #self.confidence_interval(),
            #self.power()
            )

        return self.test_result
        

    def test_statistic(self, data: Tuple[Data]):
        """
        Correct correlation type test statistic.
        """
        data1, data2 = data
        data1 = Data(data1)
        data2 = Data(data2)

        if (data1.type == "NOMINAL" and data2.type == "NOMINAL"):
            return self.matthews_coefficient(data1, data2)
        elif (data1.type == "ORDINAL" and data2.type == "NOMINAL"):
            return self.rank_biserial_coefficient(data1, data2)
        elif (data1.type == "ORDINAL" and data2.type == "ORDINAL"):
            return self.kendall_tau_b(data1, data2)
        elif (data1.type == "CONTINUOUS" and data2.type == "ORDINAL"):
            return self.spearman_rank_coefficient(data1, data2)
        elif (data1.type == "CONTINUOUS" and data2.type == "CONTINUOUS"):
            return self.pearson_coefficient(data1, data2)
        else:
            raise ValueError
    
    def print_result(self):
        """
        
        """
        return f"Correlation result goes here..."



