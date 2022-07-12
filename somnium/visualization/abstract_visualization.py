#!/usr/bin/env python

from typing import Callable, Dict, Set
from data import Variable
from visualization.figure_type import FigureType
from ..logging.ilogger import ILogger
from ivisualization import IVisualize

class AbstractVisualization(IVisualize):
    def __init__(self, logger: ILogger, file_handler, figure_type: FigureType) -> None:
        
        self.logger = logger
        self.__file_handler = file_handler
        self.__figure_type = figure_type

        self.__plot_types: Set[FigureType] = dict()
    
    def plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        if self.__is_plot_correct(group1, group2, figure_type):
            self.safe_plot(group1, group2, figure_type)

        self.__file_handler.save_figure() #TODO
        self.logger.log("Log message here...")

    def safe_plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        raise NotImplementedError()

    def __is_plot_correct(self, group1: Variable, group2: Variable, figure_type: FigureType) -> bool:
        """
        Check if figure type is correct (missing) and can be made for given data type.
        """
        return figure_type in self.__figure_type
            