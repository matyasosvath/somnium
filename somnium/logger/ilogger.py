#!/usr/bin/env python

import abc
from logger.log_level import LogLevel

class ILogger(metaclass=abc.ABCMeta):
    """
    A general interface for logging.
    All applications classes should be using one.
    """

    def log(self, level: LogLevel, message: str) -> None:
        """
        Log a single event.
        The message will be logged only if the given log level is
        configured for the logger.

        Args:
            level: The severity level of the event.</param>
            message: The message of the event.</param>
        """
        raise NotImplementedError()

