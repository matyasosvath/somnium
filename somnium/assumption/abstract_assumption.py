#!/usr/bin/env python

from typing import Dict, Iterable, Union
from assumption.inormality import INormalityTest
from assumption.ioutliertest import IOutlierTest
from result import Result
from variable import Variable
#from logging.ilogger import ILogger

import pingouin as pg
import numpy as np

class AbstractAssumption(INormalityTest, IOutlierTest):
    def __init__(self, logger = None) -> None:
        
        self.__assumptions: Dict[str, Dict[str, Union[float,bool]]] = dict()
        self.logger = logger
    
    @property
    def assumptions(self) -> Dict[str, Dict]:
        return self.__assumptions

    def normality_test(self, data: Iterable[float], name: str=None) -> Dict[str, Union[float,bool]]:
        
        self.assumptions[name] = dict()
        self.assumptions[name]["normality"] = dict()
        
        result = pg.normality(data)
        
        self.assumptions[name]["normality"]["test_statistic"] = result["W"][0]
        self.assumptions[name]["normality"]["p_value"] = result["pval"][0]
        self.assumptions[name]["normality"]["is_normal"] = result["normal"][0]

        return self.assumptions[name]["normality"]

    def has_outlier(self, data: Iterable[float]) -> Dict[str, Union[float,bool]]:
        raise NotImplementedError()

    def standardize(self, data: Iterable[float]) -> Dict[str, Union[float,bool]]:
        raise NotImplementedError()
    
    def check(self, data: Variable, name: str = None, standardize=False) -> None:
        """
        Check all assumptions and return dictionary.
        """
        self.normality_test(data, name)
        #self.has_outlier(data)
        #if standardize:
        #    self.standardize(data)
        
        #self.logger.log("INFO", "Assumptions has been checked!")

        return self.assumptions


if __name__ == '__main__':
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    actual = self.assmp.normality_test(s)
    print(actual)


