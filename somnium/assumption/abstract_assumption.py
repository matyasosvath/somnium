#!/usr/bin/env python

from typing import Dict
from assumption.inormality import INormalityTest
from assumption.ioutliertest import IOutlierTest
from test_result import TestResult
from data import Data
from ..logging.ilogger import ILogger


class AbstractAssumption(INormalityTest, IOutlierTest):
    def __init__(self, logger: ILogger=None) -> None:
        
        self.__assumptions: Dict[str, TestResult] = dict()
        self.logger = logger

    def get_assumptions(self) -> Dict[str, TestResult]:
        return self.__assumptions

    def normality_test(self, data: Data) -> TestResult:
        self.__assumptions["is_normal"] = True
        return self.__assumptions["is_normal"]

    def has_outlier(self, data: Data) -> bool:
        self.__assumptions["has_outlier"] = True
        return self.__assumptions["has_outlier"]

    def standardize(self, data: Data) -> Data:
        self.__assumptions["is_standardized"] = True
        return self.__assumptions["is_standardized"]

    def check(self, data: Data, standardize=False) -> None:
        """
        
        """
        self.normality_test(data)
        self.has_outlier(data)
        if standardize:
            self.standardize(data)
        
        #self.logger.log("INFO", "Assumptions has been checked!")

        return None

        


