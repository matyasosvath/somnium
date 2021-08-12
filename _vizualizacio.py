#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve



import os
import logging

logger = logging.getLogger()



# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "figures")
os.makedirs(IMAGES_PATH, exist_ok=True)


# Figure size
plt.figure(figsize=(14, 8))


def save_fig(fig_name, tight_layout=True, fig_extension="png", resolution=300):
    """
    Például save_fig('teszt') -> output teszt.png
    """
    path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
    #path = fig_id + "." + fig_extension
    logger.info("Saving figure")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return True 

def box_and_whiskers(df, csoportok:str, pontszamok:str):
    """
    df: dataframe
    x-tengely: egy vagy több csoport
    y-tengely: egy folytonos változó
    """
    sns.boxplot(x=csoportok, y=pontszamok, data=df)
    save_fig('box_whiskers_diagram')
    return True


def vonal_diagram():
    # line plot
    pass

def oszlopdiagram(df, csoportok: str, pontszamok: str):
    sns.barplot(x=csoportok, 
                y=pontszamok, 
                data=df, 
                estimator=np.mean, 
                ci=95, 
                capsize=0.05, 
                color='lightblue')

    save_fig('oszlopdiagram3')    

def hisztogram():
    #hist
    pass

def scatter_plot(df, x,y):
    sns.scatterplot(x=x, y=y, data=df)
    #sns.regplot(df['grade'],df['score'])
    title = x.name + "-" + y.name + "-" + "scatter-plot"
    print(title)
    save_fig(title)
    #save_fig(title)

def normality_plot(x):
    x.plot.kde()
    save_fig('normality_plot')




######################
## Machine learning ##
######################

def plot_roc_curve(y_train,y_scroes, label=None):
    # Caculate
    # True positive rate (recall) and false positive rate (1- true negative rate)
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)

    # Visualize ROC Curve
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid


import pandas as pd
import numpy as np
import pingouin as pg

df = pg.read_dataset('penguins')


scatter_plot(df, df['body_mass_g'], df['bill_length_mm'])
