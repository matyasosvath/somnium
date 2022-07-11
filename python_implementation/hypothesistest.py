
from __future__ import annotations
from typing import Iterable, Tuple
from correlation.test_result import TestResult
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
