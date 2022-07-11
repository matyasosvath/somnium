#!/usr/bin/env python



from assumption.assumption import Assumption
from correlation.correlation import Correlation
#from logging.console_logger import ConsoleLogger

import logging


x = [1,2,3,4,5,6,7,8,9]
y = [1,2,3,4,5,6,7,8,9]


#logger = ConsoleLogger()
assmp = Assumption()

corr = Correlation(assmp, x,y)

corr.correlate()