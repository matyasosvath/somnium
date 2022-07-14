#!/usr/bin/env python

from typing import Callable, Dict, Set
from file_handler import FileHandler
from variable import Variable
from visualization.figure_type import FigureType
from logger.ilogger import ILogger
from visualization.abstract_visualization import AbstractVisualization

import seaborn as sns

class Visualize(AbstractVisualization):
    def __init__(self, figure_type: FigureType, file_handler: FileHandler, logger: ILogger = None, ) -> None:
        super().__init__(logger, file_handler=file_handler, figure_type=figure_type)

    def safe_plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        """
        """
        if figure_type.SCATTER_PLOT:
            sns.scatterplot(group1.values, group2.values)
        elif figure_type.BAR_PLOT:
            sns.barplot(group1.values, group2.values)
        else:
            raise ValueError