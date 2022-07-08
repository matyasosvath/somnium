#!/usr/bin/env python

from enum import Enum
from typing import Callable
from log_level import LogLevel
from ilogger import ILogger


class AbstractLogger(ILogger):
    """
    An abstract implementation of the {@code Logger} interface.
    Holds some common logic.
    Extending classes should override the {@code safeLog} method which
    hides the filtering logic for the different log levels.
    """

    def __init__(self, levels: LogLevel):
        """
        Initializes a new instance of the AbstractLogger class.
        """

        if (levels == None):
            raise ArgumentNullException()

        self._levels = levels

    def log(self, level: LogLevel, message: str) -> None:
        """

        """
        self.__with_level(level, message)


    def safe_log(self, level: LogLevel, message: str) -> None:
        """
        Log in a safe manner.
        """
        raise NotImplementedError()

    def __with_level(self, level: LogLevel, message: str) -> None:
        """

        """
        if level in LogLevel.__members__.values():
            self.safe_log(level, message)
