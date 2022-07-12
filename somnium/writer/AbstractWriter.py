#!/usr/bin/env python3

from typing import Callable, Set
from FormatLevel import FormatLevel
from IWriter import IWriter

class AbstractWriter(IWriter):

    def __init__(self, format_level: FormatLevel, file_handler) -> None:
        self.format_level: Set[FormatLevel] = set(format_level)
        self.__file_handler = file_handler

    def write(self, format_level: FormatLevel , text: str) -> None:
        if self.with_level(format_level):
            self.safe_write(format_level, text)
        return None
    
    def safe_write(self, format_level: FormatLevel , text: str) -> None:
        raise NotImplementedError()

    def with_level(self, format_level: FormatLevel) -> bool:
        return format_level in self.format_level
            

