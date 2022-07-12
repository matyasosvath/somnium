#!/usr/bin/env python

from typing import Tuple
from test_result import TestResult
from variable import Data


class IOutlierTest(object):
    """
    
    """
    
    def __init__(self) -> None:
        pass

    def mahalanobis_distance(self, *data: Tuple[Data]) -> TestResult:
        raise NotImplementedError()