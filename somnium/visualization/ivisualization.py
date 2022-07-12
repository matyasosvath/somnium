#!/usr/bin/env python

from data import Variable
from ..logging.ilogger import ILogger

class IVisualize(object):

    def plot(self, group1: Variable, group2: Variable) -> None:
        raise NotImplementedError()