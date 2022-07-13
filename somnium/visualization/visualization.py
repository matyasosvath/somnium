#!/usr/bin/env python

from typing import Callable, Dict, Set
from variable import Variable
from visualization.figure_type import FigureType
from logging.ilogger import ILogger
from abstract_visualization import AbstractVisualization

import seaborn as sns

class Visualize(AbstractVisualization):
    def __init__(self, logger: ILogger = None, file_handler=None, figure_type: FigureType) -> None:
        super().__init__(logger, file_handler, figure_type)

    def safe_plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        """
        """
        if figure_type.SCATTER_PLOT:
            sns.scatterplot(group1.values, group2.values)
        elif figure_type.BAR_PLOT:
            sns.barplot(group1.values, group2.values)
        else:
            raise ValueError