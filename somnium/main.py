#!/usr/bin/env python



from assumption.assumption import Assumption
from correlation.correlation import Correlation
#from logging.console_logger import ConsoleLogger
from variable import Variable
import logging


x = Variable([1,2,2,3,5,6,7,1,4,5])
y = Variable([1,2,3,4,5,6,7,2,4,4])


#logger = ConsoleLogger()
assmp = Assumption()

corr = Correlation(assmp, x,y)

print(corr.correlate())