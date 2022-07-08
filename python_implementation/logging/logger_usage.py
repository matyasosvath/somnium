

import console_logger
from log_level import LogLevel as lvl


cl = console_logger.ConsoleLogger(lvl)

cl.log(lvl.INFO, "hello world")
