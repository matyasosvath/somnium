#!/usr/bin/env python

from assumption.assumption import Assumption
from correlation.correlation import Correlation
from visualization.figure_type import FigureType
from visualization.visualization import Visualize
from writer.Writer import Writer
from util.file_handler import FileHandler
from writer.format_level import FormatLevel
from util.data_handler import DataHandler
from somnium import Somnium
from logger.ilogger import ILogger


def main():
    # NOTE: Somnium-nak lehetne példányosítani valamennyi részét. 
    # Szükség van-e ezeketre a függőségekre?
    logger = ILogger()
    figure_type = FigureType
    assumption = Assumption()
    file_handler = FileHandler()
    data_handler = DataHandler(file_handler)
    visualization = Visualize(figure_type, file_handler=file_handler)
    writer = Writer(FormatLevel, file_handler)

    correlation = Correlation(assumption, visualization, writer)

    somnium = Somnium(logger, data_handler, correlation) 
    # NOTE: A somniumnak kellene-e példányosítani a correlation classt? 
    # Mert a main nem csinál semmit vele, nem ad hozzá semmit.

    somnium.run("database.xlsx")



if __name__ == '__main__':
    main()