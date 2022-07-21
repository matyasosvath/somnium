#!/usr/bin/env python

from typing import Callable, Iterable, List, Tuple
from hypothesistest import HypothesisTestPermute
from correlation.abstract_correlation import AbstractCorrelation
from assumption.assumption import Assumption
from visualization.ivisualization import IVisualize
from visualization.figure_type import FigureType
from logger.ilogger import ILogger
from writer.IWriter import IWriter
from writer.format_level import FormatLevel
from variable import Variable
from result import Result

import numpy as np


class Correlation(AbstractCorrelation):  # HypothesisTestPermute
    def __init__(self, assumption: Assumption, visualize: IVisualize, writer: IWriter, logger: ILogger = None):
        super().__init__(assumption, visualize, writer, logger)

        self.group1: Variable = None
        self.group2: Variable = None
        self.result = None
        self.correlation_name: str = ""

    def correlate(self, group1: Variable, group2: Variable) -> Result:

        self.group1, self.group2 = group1, group2
        
        # Check normality and outliers
        self.assumption.evaluate(group1.values, group1.name)
        self.assumption.evaluate(group2.values, group2.name)

        # Plot (with save plot)
        self.visualization.plot(group1, group2, FigureType.SCATTER_PLOT)

        corr = self.__decide_correlation_type() # corr.correlate() -> metódus belcsomagolva egy osztályba és ezen az osztályon belül lenne
        self.result = corr(group1.values, group2.values) # Correlation Test

        self.writer.write(FormatLevel.TEXT, self.print_result())

        return self.result

    def __decide_correlation_type(self) -> Callable:
        """
        Get correct correlation type based on data type, normality and (if applicable) outliers.

        Inherent complexity -> nem lehet tovább bontani! Nem nyerek vele semmit ebben az értelemben.
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
            self.assumption.multivariate_normality_test(
                self.group1.values, self.group2.values)
            if self.assumption.assumptions["multivariate_normality"]["is_normal"] and (self.assumption.assumptions["has_outlier"] is not True):
                self.correlation_name = "Pearson correlation"
                return self.pearson_coefficient
            else:
                self.correlation_name = "Spearman correlation"
                return self.spearman_rank_coefficient
        else:
            raise ValueError

    # region Print

    def __print_statistically_significant(self) -> str:
        significant = 'significant' if self.result["p-val"][0] <= 0.05 else 'not significant'
        r = self.result["r"][0]
        p_value = self.result["p-val"][0]
        return f'There was {significant} {self.__print_corr_negative_or_positive()} correlation between the two variables, r(df) = {r}, p = {p_value}.'

    def __print_normally_dist(self, group: Variable) -> str:
        is_normal = 'non-normal' if self.assumption.assumptions[group.name]["normality"]["is_normal"] <= 0.05 else 'normal'
        W = np.round(
            self.assumption.assumptions[group.name]["normality"]["test_statistic"], 3)
        p = np.round(
            self.assumption.assumptions[group.name]["normality"]["p_value"], 3)

        return f"{group.name} show {is_normal} distribution (W={W},p={p})."

    def __print_corr_negative_or_positive(self) -> str:
        return "positive" if self.result["r"][0] >= 0 else "negative"

    def __print_correlation(self) -> str:
        return f"{self.correlation_name} was computed to assess the linear relationship between {self.group1.name} and {self.group2.name}."

    def print_result(self) -> str:
        return f"""
            {self.__print_correlation()}
            {self.__print_normally_dist(self.group1)}
            {self.__print_normally_dist(self.group2)}
            {self.__print_statistically_significant()}
            """

    # endregion



"""
IOutput interface
gerneate output metódus

graphoutput, thesisoutput

korreláció csak generálja az ouptutokat, haladjon végig a lehetőségeken, egy listán

ha ell később egy excel output


"""