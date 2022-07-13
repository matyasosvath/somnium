#!/usr/bin/env python

from typing import Tuple
from result import Result
from variable import Variable


class INormalityTest(object):
    """
    
    """
    
    def __init__(self) -> None:
        pass

    def kolmogorov_szmirnov(self, data: Variable) -> Result:
        raise NotImplementedError()

    def shapiro_wilk(self, data: Variable) -> Result:
        raise NotImplementedError()

    def henze_zirkler(self, *data: Tuple[Variable]) -> Result:
        raise NotImplementedError()
