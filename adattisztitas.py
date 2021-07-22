#!/usr/bin/env python


import numpy as np
import pandas as pd
import logging
import pingouin
import seaborn as sns

import vizualizacio as viz


class AdatTisztitas:
    def __init__(self):
        pass

    def hianyzo_ertekek(self, df):
        sns_plot = sns.heatmap(df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
        viz.save_fig('hianyzo_ertekek', tight_layout=True)



    def run(self):
        pass

if __name__ == '__main__':
    import pingouin as pg
    df = pg.read_dataset('penguins')
    x = AdatTisztitas()
    x.hianyzo_ertekek(df)
