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
from file_handler import FileHandler
from writer.format_level import FormatLevel

from data_handler import DataHandler


class Somnium:
    def __init__(self, assumption: Assumption, visualize: IVisualize, writer: IWriter, logger: ILogger,
                file_handler: FileHandler, data_handler: DataHandler, correlation: Correlation
            ):
        
        self.assumption = assumption
        self.visualize = visualize
        self.writer = writer
        self.logger = logger
        
        self.file_handler = file_handler
        self.data_handler = data_handler

        self.correlation = correlation
    

    def run(self, name: str) -> None:

        self.data_handler.load_data(name)
        self.data_handler.create_variable_combination()

        for combination in self.data_handler.variable_combinations:
            group1 = Variable(self.data_handler.df[combination[0]], name=combination[0])
            group2 = Variable(self.data_handler.df[combination[1]], name=combination[1])

            # Correlation tests
            self.correlation.correlate(group1, group2)

if __name__ == "__main__":
    pass