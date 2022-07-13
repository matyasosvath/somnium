#!/usr/bin/env python

from typing import Iterable, List, Tuple
from hypothesistest import HypothesisTestPermute
from correlation.abstract_correlation import AbstractCorrelation
from assumption.assumption import Assumption
from visualization.ivisualization import IVisualize
#from visualization.visualization import Visualize
#from logging.console_logger  import ConsoleLogger
#from logging.ilogger import ILogger

from variable import Variable
from result import Result


class Correlation(AbstractCorrelation): # HypothesisTestPermute
    """
    
    """

    def __init__(self, assumption: Assumption, visualize: IVisualize,  group1: Variable, group2: Variable, *data: Tuple[Variable]):
        #super().__init__(data)
        self.group1 = group1
        self.group2 = group2
        
        self.assumption = assumption
        self.visualization = visualize
        #self.logger = logger

        self.result = dict()

    def correlate(self) -> Result:
        """
        Full-process
        1. Normality test
        2. Check data type
        """
        self.assumption.check(self.group1.values, self.group1.name)
        self.assumption.check(self.group2.values, self.group2.name)

        self.visualization.plot(self.group1, self.group2)

        corr = self.__decide_correlation_type()
        self.result = corr(self.group1, self.group2)
        #print(self.result)  
        self.result["corr_name"] = None
        return self.result

    def __decide_correlation_type(self):
        """
        Correct correlation type for data.
        """
        if (self.group1.type == "NOMINAL" and self.group2.type == "NOMINAL"):

            return self.matthews_coefficient
        elif ((self.group1.type == "ORDINAL" and self.group2.type == "NOMINAL") or (self.group1.type == "NOMINAL" and self.group2.type == "ORDINAL")):
            return self.rank_biserial_coefficient
        elif (self.group1.type == "ORDINAL" and self.group2.type == "ORDINAL"):
            return self.kendall_tau_b
        elif ((self.group1.type == "CONTINUOUS" and self.group2.type == "ORDINAL") or (self.group1.type == "ORDINAL" and self.group2.type == "CONTINUOUS")):
            return self.spearman_rank_coefficient
        elif (self.group1.type == "CONTINUOUS" and self.group2.type == "CONTINUOUS"):
            self.result["corr_name"] = "Pearson correlation"
            return self.pearson_coefficient
        else:
            raise ValueError

    def __is_statistically_significant(self) -> str:
        print(self.result["p-val"])
        return 'significant' if self.result["p-val"][0] <= 0.05 else 'not significant'

    def __is_normally_dist(self) -> str:
        return 'non-normal' if self.assumption.assumptions[self.group1.name]["normality"]["is_normal"] <= 0.05 else 'normal'
    
    def __is_corr_negative_or_positive(self) -> str:
        return "positive" if self.result["r"][0] >= 0 else "negative"

    def print_result(self) -> str:
        return f"""
            {self.result["corr_name"]} was computed to assess the linear relationship
            between {self.group1.name} and {self.group2.name}.
            
            {self.group1.name.upper()} show {self.__is_normally_dist()} distribution 
            (W={self.assumption.assumptions[self.group1.name]["normality"]["test_statistic"]},p={self.assumption.assumptions[self.group1.name]["normality"]["p_value"]}).
            {self.group2.name.upper()} show {self.__is_normally_dist()} distribution 
            (W={self.assumption.assumptions[self.group2.name]["normality"]["test_statistic"]},p={self.assumption.assumptions[self.group2.name]["normality"]["p_value"]}).
            
            There was {self.__is_statistically_significant()} {self.__is_corr_negative_or_positive()} correlation between the two variables, 
            r(df) = {self.result["r"][0]}, p = {self.result["p-val"][0]}.
            """
