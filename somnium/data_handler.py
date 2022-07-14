#!/usr/bin/env python

import pandas as pd
import numpy as np
from variable import Variable
from typing import Tuple

class DataHandler:
    def __init__(self, name: str) -> None:
        self.df: pd.DataFrame = self.load_data(name)

    @classmethod
    def load_data(self, df_name: str, **kwargs):
        return pd.read_excel(df_name)
        

    @classmethod
    def create_variable_combination(self) -> Tuple[Tuple[Variable]]:
        """
        Create tuples of variable combinations
        """
        pass
