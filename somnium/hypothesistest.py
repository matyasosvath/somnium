
from variable import Variable
from typing import Iterable, List, Tuple
from result import Result
import numpy as np
import random

class HypothesisTestPermute(object):

    def __init__(self, data: Tuple[Variable]):
        self.n: int
        self.m: int
        self.data: Tuple[Variable] = data
        self.pool: List[float]
        self.__test_stats: Iterable[float]

        self.__make_model()
        self.actual = self.test_statistic(data)

    def __permutate_test_statistics(self, iters: int):
        self.__test_stats = [self.test_statistic(self.__run_model()) for _ in range(iters)]

    def p_value(self, iters: int = 1000) -> float:
        """
        Calculate p-value.
        """
        self.__permutate_test_statistics(iters)

        count = 0
        self.__type_2_error = 0
        for x in self.__test_stats:
            if x >= self.actual:
                count +=1
            else:
                self.__type_2_error +=1
        #count = sum(1 for x in self.__test_stats if x >= self.actual)
        return count / iters

    def confidence_interval(self, ci_level: int = 95) -> Tuple[float]:
        """
        Bootstrapped Confidence Interval (CI).
        
        It is the interval that encloses the central 90% of the bootstrap 
        sampling distribution of a sample statistic. 

        More generally, an x% confidence interval around a sample estimate should, 
        on average, contain similar sample estimates x% of the time 
        (when a similar sampling procedure is followed).
        """
        
        sorted_test_statistics = sorted(self.__test_stats)

        # Trim endpoints of resampled CI
        trim = ((1 - (ci_level/100))/2)
        print(f"Trim: {trim}")
        endpoints = int(trim*100)
        print(f"Endpoint: {endpoints}")
        trimmed_ci = sorted_test_statistics[endpoints:-endpoints]
        lower, upper = min(trimmed_ci), max(trimmed_ci)
        return (lower, upper)

    def power(self) -> float:
        """
        The probability of detecting a given effect size with a given sample size.
        """
        return 1000 / self.__type_2_error

    def __make_model(self) -> None:
        """
        Creates a pool from the two groups.
        """
        group1, group2 = self.data
        self.n, self.m = group1.N, group2.N
        self.pool = group1.values + group2.values

    def __run_model(self) -> Variable:
        """
        Shuffle pool (randomisation).
        Create two groups (with length of group 1 and 2).
        """
        random.shuffle(self.pool)
        data1 = Variable(self.pool[:self.n])
        data2 = Variable(self.pool[self.n:])
        data = data1, data2
        #data = self.pool[:self.n], self.pool[self.n:]

        return data

    def test_statistic(self, *data: Tuple[Variable]) -> float: 
        raise NotImplementedError()



if __name__ == '__main__':
    pass