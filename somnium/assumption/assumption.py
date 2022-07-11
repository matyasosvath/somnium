#!/usr/bin/env python

from typing import List
from collections.abc import Iterable
from .abstract_assumption import AbstractAssumption

#from logging.ilogger import ILogger
#from logging.console_logger import ConsoleLogger

class Assumption(AbstractAssumption):
    def __init__(self, logger=None) -> None:
        super().__init__(logger)