#!/usr/bin/env python

from typing import Callable
from log_level import LogLevel
from abstract_logger import AbstractLogger
import datetime


class ConsoleLogger(AbstractLogger):
    """
    An implementation of the ILogger interface.
    Logs messages for the standard error stream of the process.
    """

    def safe_log(self, level: LogLevel, message: str) -> None:
        """
        Log in a safe manner.
        """
        formatted_message = self.__format_message(level, message)
        print(formatted_message)

    def __format_message(self, level: LogLevel, message: str) -> str:
        time = datetime.datetime.now()
        return f"{time} - {level.name}: {message}"


