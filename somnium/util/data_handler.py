#!/usr/bin/env python

import pandas as pd
import numpy as np
from util.file_handler import FileHandler
from variable import Variable
from typing import Tuple
import itertools


class DataHandler:
    def __init__(self, file_handler: FileHandler) -> None:
        self.file_handler  = file_handler

        self.df: pd.DataFrame = None
        self.variable_combinations: Tuple[Tuple[Variable]] = None


    def load_data(self, name:str):
        self.df = self.file_handler.load_data(name)
        return self.df
        
    def create_variable_combination(self) -> Tuple[Tuple[Variable]]:
        """
        Create tuples of variable combinations
        """
        self.variable_combinations = tuple(itertools.combinations(self.df.columns, 2))
        return self.variable_combinations



if __name__ == '__main__':
    fh = FileHandler()
    dh = DataHandler(fh)
    dh.load_data("database.xlsx")
    combs = dh.create_variable_combination()
    print(combs)
