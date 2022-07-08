#!/usr/bin/env python

from enum import Enum, auto


class Data:
    def __init__(self, values):
        self.values = values
        self.type = self.get_data_type(values)
        self.n = len(values)

    def get_data_type(self, values):
        """
        Get given variable data type.
        """
        unique_values_number = len(set(values))

        if unique_values_number < 2: return DataType.NOMINAL.name
        if unique_values_number < 7: return DataType.ORDINAL.name

        return DataType.CONTINUOUS.name

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __str__(self):
        return f""

    def __repr__(self):
        return self.values


class DataType(Enum):
    NOMINAL = auto()
    ORDINAL = auto()
    CONTINUOUS = auto()

