#!/usr/bin/env python

from typing import Callable
from log_level import LogLevel
from logger_interface import ILogger


class AbstractLogger(ILogger):
    """
    An abstract implementation of the {@code Logger} interface.
    Holds some common logic.
    Extending classes should override the {@code safeLog} method which
    hides the filtering logic for the different log levels.
    """
    _levels = set()


    def __init__(self, levels: LogLevel):
        """
        Initializes a new instance of the AbstractLogger class.
        """
        if (levels == null):
            raise ArgumentNullException()

        _levels = set(log_level)

    def log(level: LogLevel, message: string) -> None:
        """

        """
        with_level(level, _safe_log(level, message))


    def _safe_log(level: LogLevel, message: string) -> None:
        """

        """
        raise NotImplementedError()

    def _with_level(level: LogLevel, func: Callable[[LogLevel, string], None]) -> None:
        """

        """
        if level in _levels:
            func()

