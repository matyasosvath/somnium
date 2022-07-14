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

x = Variable([1,2,2,3,5,6,7,1,4,5], name="Test1")
y = Variable([1,2,3,4,5,6,7,2,4,4], name="Test2")

figure_type = FigureType
assmp = Assumption()
file_handler = FileHandler()
vis = Visualize(figure_type, file_handler=file_handler)
writer = Writer(FormatLevel, file_handler)

corr = Correlation(assmp, vis, writer)

print(corr.correlate(x,y))
print(corr.print_result())

fh = FileHandler()
dh = DataHandler(fh)
df = dh.load_data("database.xlsx")
combs = dh.create_variable_combination()

for comb in combs:
    x = Variable(df[comb[0]], name=comb[0])
    y = Variable(df[comb[1]], name=comb[1])
    print(x, x.name)
    print(y, y.name)

    print(corr.correlate(x,y))
    print(corr.print_result())
