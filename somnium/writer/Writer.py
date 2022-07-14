#!/usr/bin/env python3

from file_handler import FileHandler
from writer.format_level import FormatLevel
from writer.AbstractWriter import AbstractWriter


class Writer(AbstractWriter):
    def __init__(self, format_level: FormatLevel, file_handler: FileHandler) -> None:
        super().__init__(format_level, file_handler)
    
    def safe_write(self, format_level: FormatLevel, text: str) -> None:
        self.file_handler.write(self.format_text(format_level, text))
    
    def format_text(self, format_level: FormatLevel, text: str) -> str:
        if format_level == self.format_level.HEADING:
            return f"# {text}"
        elif format_level == self.format_level.TEXT:
            return f"{text}"
        elif format_level == self.format_level.MATH:
            return f"$${text}$$"
        else:
            raise ValueError