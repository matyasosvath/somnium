

import console_logger
from log_level import LogLevel as lvl


logger = console_logger.ConsoleLogger()

logger.log(lvl.INFO, "hello world")
