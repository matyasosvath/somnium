#!/usr/bin/env python


import numpy as np
import pandas as pd
import logging



class HipotezisTesztek:
    """
    Összes hipotézis teszt leírása és kódja itt.
    """
    def __init__(self):
        pass




### TwO GROUPS



### Kettő vagy több csoportra ellenőrzése
def ketto_vagy_tobb_csoport_ellenorzese(df, groups):
    """
    ha kettőnél több csoport van -> return true
    kettő csoport van ->return False
    """
    return df[groups].nunique() > 2


### MULTIPLE GROUPS

def anova_one_way():
    pass

def welch_f_test():
    pass


def kruskal_wallis():
    pass
