#!/usr/bin/env python3

from writer.format_level import FormatLevel

class IWriter:
    def write(self, format_level: FormatLevel , text: str):
        raise NotImplementedError()
    

