#!/usr/bin/env python

from typing import Tuple
from test_result import TestResult
from data import Data


class INormalityTest(object):
    """
    
    """
    
    def __init__(self) -> None:
        pass

    def kolmogorov_szmirnov(self, data: Data) -> TestResult:
        raise NotImplementedError()

    def shapiro_wilk(self, data: Data) -> TestResult:
        raise NotImplementedError()

    def henze_zirkler(self, *data: Tuple[Data]) -> TestResult:
        raise NotImplementedError()
