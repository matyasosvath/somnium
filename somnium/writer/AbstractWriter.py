#!/usr/bin/env python3

from typing import Callable, Set
from util.file_handler import FileHandler
from writer.format_level import FormatLevel
from writer.IWriter import IWriter

class AbstractWriter(IWriter):

    def __init__(self, format_level: FormatLevel, file_handler: FileHandler) -> None:
        self.format_level: FormatLevel = format_level
        self.file_handler: FileHandler = file_handler

    def write(self, format_level: FormatLevel , text: str) -> None:
        if self.with_level(format_level):
            self.safe_write(format_level, text)
        return None
    
    def safe_write(self, format_level: FormatLevel , text: str) -> None:
        raise NotImplementedError()

    def with_level(self, format_level: FormatLevel) -> bool:
        return format_level in set(self.format_level)
            

