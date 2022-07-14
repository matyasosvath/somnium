#!/usr/bin/env python

from assumption.assumption import Assumption
from correlation.correlation import Correlation
from logger.console_logger import ConsoleLogger
from variable import Variable
from visualization.figure_type import FigureType
from visualization.visualization import Visualize
from writer.Writer import Writer
from file_handler import FileHandler
from writer.format_level import FormatLevel
from data_handler import DataHandler
from somnium import Somnium
from logger.ilogger import ILogger


def main():
    logger = ILogger()
    figure_type = FigureType
    assumption = Assumption()
    file_handler = FileHandler()
    data_handler = DataHandler(file_handler)
    visualization = Visualize(figure_type, file_handler=file_handler)
    writer = Writer(FormatLevel, file_handler)

    correlation = Correlation(assumption, visualization, writer)

    somnium = Somnium(assumption, visualization, writer, logger,
                    file_handler, data_handler, correlation)

    somnium.run("database.xlsx")



if __name__ == '__main__':
    main()