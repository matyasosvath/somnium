#!/usr/bin/env python



from assumption.assumption import Assumption
from correlation.correlation import Correlation
#from logging.console_logger import ConsoleLogger
from variable import Variable
import logging


x = Variable([1,2,3,4,5,6,7,8,9,3,3,2,1,2,4,5,6,7])
y = Variable([1,2,3,4,5,6,7,8,9,3,4,5,6,7,7,5,1,2])


#logger = ConsoleLogger()
assmp = Assumption()

corr = Correlation(assmp, x,y)

print(corr.correlate())