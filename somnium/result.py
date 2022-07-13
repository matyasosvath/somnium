#!/usr/bin/env python

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Result:
    n: int
    test_statistic: float
    p_value: float
    ci: Tuple[float]
    power: float