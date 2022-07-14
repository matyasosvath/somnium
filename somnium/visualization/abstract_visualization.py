#!/usr/bin/env python

from typing import Callable, Dict, Set
from file_handler import FileHandler
from variable import Variable
from visualization.figure_type import FigureType
from logger.ilogger import ILogger
from visualization.ivisualization import IVisualize

class AbstractVisualization(IVisualize):
    def __init__(self, logger: ILogger, file_handler: FileHandler, figure_type: FigureType) -> None:
        
        self.logger: ILogger = logger
        self.__file_handler: FileHandler = file_handler
        self.__figure_type: FigureType = figure_type

        self.__plot_types: Set[FigureType] = dict()
    
    def plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        if self.__is_plot_correct(group1, group2, figure_type):
            self.safe_plot(group1, group2, figure_type)
            self.__file_handler.save_figure(f"{group1.name}-{group2.name}")

        #self.logger.log()

    def safe_plot(self, group1: Variable, group2: Variable, figure_type: FigureType) -> None:
        raise NotImplementedError()

    def __is_plot_correct(self, group1: Variable, group2: Variable, figure_type: FigureType) -> bool:
        """
        Check if figure type is correct (missing) and can be made for given data type.
        """
        return figure_type in self.__figure_type
            