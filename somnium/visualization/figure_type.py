#!/usr/bin/env python

from enum import Enum, auto

class FigureType(Enum):
    SCATTER_PLOT = auto()
    BOX_WHISKERS = auto()
    LINE_PLOT = auto()
    BAR_PLOT = auto()
    HISTOGRAM = auto()
    NORMALITY_PLOT = auto()
