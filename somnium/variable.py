#!/usr/bin/env python

from enum import Enum, auto
from typing import Any, Union


class Variable(object):

    # Preventing dynamic attribute assignment (https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init)
    __slots__ = ['__type', '__N', '__values', '__name']

    def __init__(self, values, name=None):
        self.__values = list(values)
        self.__type = self.__get_data_type(values)
        self.__N = len(values)
        self.__name = name

    @property
    def values(self): return self.__values

    @property
    def type(self): return self.__type

    @property
    def N(self): return self.__N

    @property    
    def name(self) -> Union[str, None]:
        if self.__name is not None:
            return self.__name        
        return None

    def __get_data_type(self, values):
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

    # def __repr__(self):
    #     return self.values
    
    def __call__(self):
        return self.values

    # def __setattr__(self, name: str, value) -> None:
    #     if hasattr(self, name):
    #         raise ValueError("Hello there!")
    #     object.__setattr__(self, name, value)

class DataType(Enum):
    NOMINAL = auto()
    ORDINAL = auto()
    CONTINUOUS = auto()


# if __name__ == '__main__':
#     d = Variable([1,2,3,4])
#     print(d)
#     print(type(d))
#     print(d[0] + d[1])