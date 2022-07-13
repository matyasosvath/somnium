#!/usr/bin/env python

from typing import Tuple
from result import Result
from variable import Variable


class IOutlierTest(object):
    """
    
    """
    
    def __init__(self) -> None:
        pass

    def mahalanobis_distance(self, *data: Tuple[Variable]) -> Result:
        raise NotImplementedError()