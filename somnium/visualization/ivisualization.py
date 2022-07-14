#!/usr/bin/env python


from variable import Variable
from visualization.figure_type import FigureType

class IVisualize(object):

    def plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        pass