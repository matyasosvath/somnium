#!/usr/bin/env python

from argparse import ArgumentError
from typing import List, Tuple
from collections.abc import Iterable
from hypothesistest import HypothesisTestPermute
from abstract_correlation import AbstractCorrelation
from assumption.assumption import Assumption

from logging.console_logger import ConsoleLogger
from logging.ilogger import ILogger

from data import Data
from test_result import TestResult


class Correlation(HypothesisTestPermute, AbstractCorrelation):
    """
    
    """

    def __init__(self, assumption: Assumption, logger: ILogger, *data):
        
        super().__init__(data)

        self.assumption = assumption
        self.logger = logger

    def correlate(self, remove_outliar=False) -> TestResult:
        """
        Full-process
        1. Remove outliers
        2. Normality test
        3. Data type check
        """
        if remove_outliar:
            self.data1 = self.assumption.remove_outliar(self.data1)
            self.data2 = self.assumption.remove_outliar(self.data2)

        self.__assumption_result = dict() #assumption.check(data1, data2) #TODO missing


        self.test_result = TestResult(
            len(self.pool), 
            self.actual, 
            self.p_value(),
            self.confidence_interval(),
            self.power()
            )

        return self.test_result
        

    def test_statistic(self, *data: Tuple[Data]):
        """
        Correct correlation type test statistic.
        """
        data1, data2 = data

        if (data1.type == "NOMINAL" and data2.type == "NOMINAL"):
            return self.matthews_coefficient(data1, data2).test_statistic
        elif (data1.type == "ORDINAL" and data2.type == "NOMINAL"):
            return self.rank_biserial_coefficient(data1, data2).test_statistic
        elif (data1.type == "ORDINAL" and data2.type == "ORDINAL"):
            return self.kendall_tau_b(data1, data2).test_statistic
        elif (data1.type == "CONTINUOUS" and data2.type == "ORDINAL"):
            return self.spearman_rank_coefficient(data1, data2).test_statistic
        elif (data1.type == "CONTINUOUS" and data2.type == "CONTINUOUS"):
            return self.pearson_coefficient(data1, data2).test_statistic
        else:
            raise ValueError

    
    def print_result(self):
        """
        
        """
        pass




class Korrelacio(Assumptions, HipotezisTesztek):
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.data = pd.concat([x,y], axis=1) # két külön series-ből egy datafram legyen a "multivariate test" kedvéért
        # kell hogy a x.name-nek neve legyen különben hibát jelez ki
      

        self.x_adattipus = adattipus(self.x)
        self.y_adattipus = adattipus(self.y)
        #print(self.x_adattipus, self.y_adattipus)

        self.assumptions = self.test_for_assumptions(self.data, method="correlation")
        self.assumptions['Outliers'] = check_for_multivariate_outliers(self.data)

        self.writer = Luhmann()

        logger.info("Korrelacio class successfully initialized")
    
    def run(self):  

        # Remove outliers
        self.removed_outliers = remove_multivariate_outliers(self.data)

        #print(f"removed outliers az ez: {self.removed_outliers}")
        
        logger.info("Outliers successfully removed!")
        
        self.x, self.y = self.removed_outliers.iloc[:, 0], self.removed_outliers.iloc[:, 1]
        # print(type(self.x), type(self.y))
        # print(self.x)
        # print(self.y)

        # Folytonos-folytonos
        if self.y_adattipus == 'ratio-interval' and self.x_adattipus == 'ratio-interval':
            
            # Vizualizacio
            scatter_plot(df, x=self.x,y=self.y)
        
            if self.assumptions['Normality Test']['Henze-Zirkler']['P-value'] > 0.05 and self.assumptions['Outliers']['Multivariate Outliers'] == False:

                self.pearson(self.x,self.y)
                logger.info("Pearson test successfully run!")

            elif self.assumptions['Normality Test']['Henze-Zirkler']['P-value'] <= 0.05 or self.assumptions['Outliers']['Multivariate Outliers']:

                #print(self.x.ndim, self.y.ndim)
                self.spearman(self.x,self.y)#if self.assumptions["Univariate Outliers"]:
                logger.info("Spearman test successfully run!")

                # Bivariate Outliers
                if self.assumptions["Outliers"]['Multivariate Outliers']: 
                    self.skipped_spearman_correlation(self.x,self.y)
                    self.shepherd_pi_correlation(self.x,self.y)
                    logger.info("Skipped spearman and shepherd pi corr test ran successfully!")         
                
                # Univariate outliers
                # else:
                #     self.biweight_correlation(self.x,self.y)    
                #     self.percentage_bend_correlation(self.x,self.y)

            else:
                raise ValueError
        
        # Folytonos-Ordinális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'ratio-interval':
            # Spearman rankkorrelacio
            self.spearman(self.x,self.y)
            logger.info("Spearman test successfully run!")

        # Folytonos-Nominális
        elif self.y_adattipus == 'ratio-interval' and self.x_adattipus == 'ordinal':
            # Point-biserial correlation coefficient
            self.point_biserial_correlation(self.x,self.y)
            logger.info("Point biserial correlation test successfully run!")
            
        # Ordinális-Ordinális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'ordinal':
            # Kendalls rank correlation coefficient
            self.kendall_tau(self.x,self.y)
            logger.info("Kendall tau-b test successfully run!")

        # Ordinális-Nominális
        elif self.y_adattipus == 'ordinal' and self.x_adattipus == 'nominal':
            # Rank-biserial correlation coefficient
            self.rank_biserial_correlation(self.x,self.y)
            logger.error("Rank biserial correlation are not implemented yet!")

        # Nominális-nominális
        elif self.y_adattipus == 'nominal' and self.x_adattipus == 'nominal':
            # Phi coefficient or Matthews correlation coefficient 
            self.phi_coeff_matthews_coeff(self.x,self.y)
            logger.info("Matthews coefficient test successfully run!")

        else:
            raise ValueError("Hiba. Nincs tobb korrelacio lehetoseg.")

