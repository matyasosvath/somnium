#!/usr/bin/env python

from enum import Enum, auto
from typing import Any
from sklearn.neighbors import VALID_METRICS


class Variable(object):

    # Preventing dynamic attribute assignment
    __slots__ = ['__type', '__N', '__values', '__name']

    def __init__(self, values, name=None):
        self.__values = values
        self.__type = self.get_data_type(values)
        self.__N = len(values)
        self.__name = name

    @property
    def values(self): return self.__values

    @property
    def type(self): return self.__type

    @property
    def N(self): return self.__N

    @property    
    def name(self):
        if self.__name is not None:
            return self.__name
        raise ValueError("Name is not specified!")

    def get_data_type(self, values):
        """
        Get variable data type.
        """
        unique_values_number = len(set(values))

        if unique_values_number <= 2: 
            return DataType.NOMINAL.name
        if unique_values_number < 7: 
            return DataType.ORDINAL.name

        return DataType.CONTINUOUS.name

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __str__(self):
        return f"{self.values}"

    def __repr__(self):
        return self.values

    # def __setattr__(self, name: str, value) -> None:
    #     if hasattr(self, name):
    #         raise ValueError("Hello there!")
    #     object.__setattr__(self, name, value)

class DataType(Enum):
    NOMINAL = auto()
    ORDINAL = auto()
    CONTINUOUS = auto()



if __name__ == "__main__":
    d = Variable([10,1,1,1,1,1], name="test")
    print(d)
    print(d.type)
    #print(d.__type)
    print(d.N)
    #d.type = DataType.NOMINAL
    print(type(d.type))
    d.random = 5
    print(f"random: {d.random}")
    print(d[5])
    print(d.__dict__)