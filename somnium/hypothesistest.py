
from __future__ import annotations
from typing import Iterable, Tuple
from test_result import TestResult
import numpy as np



class HypothesisTestPermute(object):

	def __init__(self, data):
		self.n: int
		self.m: int
		self.data: Tuple[Iterable[int], Iterable[int]] = data
		self.pool: Iterable[float]
		self.__test_stats : Iterable[float]

		self.__make_model()
		self.actual = self.test_statistic(data)


	def __permutate_test_statistics(self, iters):
		self.__test_stats = [self.test_statistic(self.__run_model()) for _ in range(iters)]


	def p_value(self, iters=1000) -> float:
		"""
		Calculate p-value.
		"""
		self.__permutate_test_statistics(iters)

		count = sum(1 for x in self.__test_stats if x >= self.actual)
		return count / iters


	def confidence_interval(self): raise NotImplementedError()
	def power(self): raise NotImplementedError()

	def get_test_result(self) -> TestResult:
		raise NotImplementedError()

	def __make_model(self):
		"""
		Creates a pool from the two groups.
		"""
		print(self.data)
		group1, group2 = self.data
		self.n, self.m = len(group1), len(group2)
		self.pool = np.hstack((group1, group2))


	def __run_model(self):
		"""
		Shuffle pool (randomisation).
		Create two groups (with length of group 1 and 2).
		"""
		np.random.shuffle(self.pool)
		data = self.pool[:self.n], self.pool[self.n:]
		return data


	def test_statistic(self, *data): raise NotImplementedError() # 2 groups

	def __clear_cache(self):
		raise NotImplementedError()


#######################
### HIPOTEZIS TESZT ###
#######################


class HipotezisTesztek:

    def __init__(self):

        self.writer = Luhmann()

    logger.info("Hipotezis tesztek successfully initialized")

    def __permutation(self,v1,v2):
        """
        Permtutation with two groups
        """
        #print('permut')
        v1 = pd.Series(v1)
        v2 = pd.Series(v2)
        # shuffle groups
        data = pd.concat([v1,v2])
        data = shuffle(data)
        # resample groups
        v1 = self.__resample(data, size=len(v1), replace=False)
        v2 = self.__resample(data, size=len(v2), replace=False)
        return v1, v2

    def __resample(self, x, size, replace = False):
        """
        Bootstrap Resampling
        """
        return np.random.choice(x, size=size, replace=replace)

    def __pvalue(self, x,y, hypothesis_test, iter = 1000, ci=True, ci_level=95):
        """
        P-value
        """
        actual = hypothesis_test(x,y)
        #print(actual)

        #permute_dist = [phi_coeff_matthews_coeff(permutation(x,y)) for _ in range(iter)] # return value in permutation(x,y) do not works, dont know why
        permute_dist = []
        for _ in range(iter):
            a,b = self.__permutation(x,y)
            permute_dist.append(hypothesis_test(a,b))
        #print(permute_dist)

        # Bootstraped [bs] Confidence Interval
        if ci:
            statistics = sorted(permute_dist)
            # Trim endpoints of resampled CI
            trim = ((1 - (ci_level/100))/2)
            endpoints = int(trim*1000)
            trimmed_ci = statistics[endpoints:-endpoints]
            lower, upper = min(trimmed_ci), max(trimmed_ci)

        # Calculate bootrstrapped p-value
        count = sum(1 for i in permute_dist if i >= actual)
        # Return p-value, lower CI, upper CI
        return count/iter, lower, upper

    def __power(self, x,y, num_runs=101):
        #x, y = self.data
        power_count = 0

        for i in range(num_runs):
            resample_x = np.random.choice(x, len(x), replace=True)
            resample_y = np.random.choice(y, len(y), replace=True)

            p = self.pvalue()

            if p < 0.05:
                power_count += 1

            return power_count/num_runs

