#!/usr/bin/env python


import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

import logging


## Sajat modulok
from _vizualizacio import *


logger = logging.getLogger()

def beolvaso(table, method='csv'):
    try:
        if method == 'excel':
            adat = pd.read_excel(table)
        if method == 'csv':
            adat = pd.read_csv(table)
    except IndexError:
        pass
        # look for all errors you ran across previously
    except ValueError:
        pass
    except ImportError:
        pass

    print(adat.head())
    return adat





class AdatTisztitas:
    def __init__(self):
        pass

    logger.info("Adattisztitas successfully initialized")


    def hianyzo_ertekek(self, df):
        sns_plot = sns.heatmap(df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
        save_fig('hianyzo_ertekek', tight_layout=True)



    def run(self):
        pass

if __name__ == '__main__':
    import pingouin as pg
    df = pg.read_dataset('penguins')
    x = AdatTisztitas()
    x.hianyzo_ertekek(df)
    box_and_whiskers(df,'species', 'body_mass_g')
    oszlopdiagram(df, 'species', 'body_mass_g')
