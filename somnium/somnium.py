#!/usr/bin/env python

from correlation.abstract_correlation import AbstractCorrelation
from assumption.assumption import Assumption
from correlation.correlation import Correlation
from correlation.icorrelation import ICorrelation

from logger.ilogger import ILogger
from logger.console_logger import ConsoleLogger

from variable import Variable

from visualization.figure_type import FigureType

from visualization.ivisualization import IVisualize
from visualization.visualization import Visualize

from writer.IWriter import IWriter
from writer.Writer import Writer
from writer.format_level import FormatLevel

from util.data_handler import DataHandler


class Somnium:
    def __init__(self, logger: ILogger, data_handler: DataHandler, correlation: Correlation
            ):
        
        self.logger = logger
        
        self.data_handler = data_handler

        self.correlation = correlation
    

    def run(self, name: str) -> None:

        self.data_handler.load_data(name)
        self.data_handler.create_variable_combination()
        #NOTE: Create util.py, a data_handler feladatán tulnyulik a kombináció képzés

        for combination in self.data_handler.variable_combinations: # (A,B)
            group1 = Variable(self.data_handler.df[combination[0]], name=combination[0])
            group2 = Variable(self.data_handler.df[combination[1]], name=combination[1])

            # Correlation tests
            self.correlation.correlate(group1, group2)

if __name__ == "__main__":
    pass