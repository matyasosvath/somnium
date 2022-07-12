#!/usr/bin/env python3

from FormatLevel import FormatLevel

class IWriter:
    def write(self, format_level: FormatLevel , text: str):
        raise NotImplementedError()
    

