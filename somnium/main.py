#!/usr/bin/env python



from assumption.assumption import Assumption
from correlation.correlation import Correlation
#from logging.console_logger import ConsoleLogger
from variable import Variable
import logging
from visualization.figure_type import FigureType
from visualization.visualization import Visualize
from writer.Writer import Writer
from file_handler import FileHandler
from writer.format_level import FormatLevel


x = Variable([1,2,2,3,5,6,7,1,4,5], name="Test1")
y = Variable([1,2,3,4,5,6,7,2,4,4], name="Test2")

figure_type = FigureType
#logger = ConsoleLogger()
assmp = Assumption()
file_handler = FileHandler()
vis = Visualize(figure_type, file_handler=file_handler)
writer = Writer(FormatLevel, file_handler)

corr = Correlation(assmp, vis, writer, x,y)

print(corr.correlate())
print(corr.print_result())