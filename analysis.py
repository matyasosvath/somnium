#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys, os
import json


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


def elemszam(adat):
    print(adat.shape[0])
    return adat.shape[0]

def adattipuskkezelo(adat):
    adattipusok = {}
    for col in adat.columns:
        a = len(adat[col].unique().tolist())
        if a <= 2:
            adattipusok[col] = 'nominal'
        if 2 < a <= 8:
            adattipusok[col] = 'ordinal'
        if a > 8:
            adattipusok[col] = 'ratio-or-interval'
    #print(f'{adattipusok} \n')
    with open('jellemzok.json', 'w') as f:
        json.dump(adattipusok, f)
    return adattipusok


# Define a szakdolgozat.md
def iras(sima_text):
    with open('szakdolgozat.md', 'a') as f:
        f.write(sima_text)
        f.write('\n \n')


# Check for a given parameter in jellemzok.json
def check(col, mit_akarok_csekkolni, hol):
    with open('jellemzok.json', 'r') as f:
        json_object_data = json.load(f)
        if mit_akarok_csekkolni in json_object_data[col]:
            return True

# Descriptive Statistics

## Interval-ratio scale
def descriptive_interval(col):

    #if == 'ratio-or-interval'
    atlag = np.round(col.mean())
    variancia = np.round(col.var())
    szoras = np.round(col.std())

    iqr_25 = np.round(col.quantile(0.25))
    iqr_75 = np.round(col.quantile(0.75))
    iqr = iqr_75 - iqr_25

    maximum = np.round(col.max())
    minimum = np.round(col.min())

    leiro_stat ={
        'atlag': atlag,
        'variancia': variancia,
        'szoras': szoras,
        'iqr': iqr,
        'max': maximum,
        'min': minimum,
    }
    return leiro_stat



## Ordinal scale
def descriptive_ordinal(col):
    pass

## Nominal scale
def descriptive_nominal(col):
    pass




# Distribution

def distribution(col):
    skewness = col.skew()
    kurtosis = col.kurtosis()
    return skewness, kurtosis




def main():
    # Get data
    adat = sys.argv[1]
    adat = beolvaso(adat)
    adattipus = adattipuskkezelo(adat)
    print(adattipus)

    minta = elemszam(adat)
    iras(f'A mintat {minta} fő alkotta.')
    
    elsocol = descriptive_interval(adat['age'])
    print('hello')
    print(f'hello my name is {elsocol}')
    print(f'hello my name is {elsocol["atlag"]}')

    iras(f'A vizsgalati szemelyek atlageletkora (és szorasa) {elsocol["atlag"]} ({elsocol["szoras"]}) ev volt.) ')






""" # Make directory for output
try:
    os.mkdir('szakdolgozat')
except FileExistsError:
    pass
os.chdir('szakdolgozat')
# print(os.getcwd())
# df.to_csv('szakdolgozat.csv', index=False)
 """


if __name__ == '__main__':
    main()



