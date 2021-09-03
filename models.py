#!/usr/bin/env python

import csv

import tesztek

class Valtozo:
    def __init__(self, ertekek):
        self.ertekek = pd.Series(ertekek)
        self.tipus = tesztek.adattipus(ertekek)
        #self.mean = mean()
        #self.std = standard_deviation()
        #self.var = variance()
        #self.se = None

    #def __str__(self):
    #    print('hello')

    #def __repr__(self):
    #    pass



if __name__ == '__main__':
    lista = [1,2,3,4,1,2,34,4,2,2]
    v = Valtozo(lista)
    print(v)
    print(v.ertekek)
    print(v.tipus)
