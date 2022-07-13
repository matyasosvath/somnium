#!/usr/bin/env python

from typing import Dict
from assumption.inormality import INormalityTest
from assumption.ioutliertest import IOutlierTest
from result import Result
from variable import Variable
#from logging.ilogger import ILogger


class AbstractAssumption(INormalityTest, IOutlierTest):
    def __init__(self, logger = None) -> None:
        
        self.__assumptions: Dict[str, Result] = dict()
        self.logger = logger

    def get_assumptions(self) -> Dict[str, Result]:
        return self.__assumptions

    def normality_test(self, data: Variable) -> Result:
        self.__assumptions["is_normal"] = True
        return self.__assumptions["is_normal"]

    def has_outlier(self, data: Variable) -> bool:
        self.__assumptions["has_outlier"] = True
        return self.__assumptions["has_outlier"]

    def standardize(self, data: Variable) -> Variable:
        self.__assumptions["is_standardized"] = True
        return self.__assumptions["is_standardized"]

    def check(self, data: Variable, standardize=False) -> None:
        """
        
        """
        self.normality_test(data)
        self.has_outlier(data)
        if standardize:
            self.standardize(data)
        
        #self.logger.log("INFO", "Assumptions has been checked!")

        return None

        


