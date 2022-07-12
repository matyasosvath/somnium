#!/usr/bin/env python3

from typing import Callable, Set
from FormatLevel import FormatLevel
from AbstractWriter import AbstractWriter

class Writer(AbstractWriter):
    def __init__(self, format_level: FormatLevel, file_handler) -> None:
        super().__init__(format_level, file_handler)
    
    def safe_write(self, format_level: FormatLevel, text: str) -> None:
        self.__file_handler.open(self.__format_text(format_level, text))
    
    def __format_text(self, format_level: FormatLevel, text: str) -> str:
        if format_level == self.format_level.HEADING:
            return f"# {text}"
        elif format_level == self.format_level.TEXT:
            return f"{text}"
        elif format_level == self.format_level.MATH:
            return f"${text}$"
        else:
            raise ValueError