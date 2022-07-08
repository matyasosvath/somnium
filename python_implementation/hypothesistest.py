
from __future__ import annotations
import numpy as np

class HypothesisTestPermute(object):

	def __init__(self, data):
		self.n: int
		self.m: int
		self.data = data #: tuple[int, int]
		self.pool #: list[int]
		self.__test_stats # :list[float]

		self.__MakeModel()
		self.actual = self.TestStatistic(data)

	def __permutate_test_statistics(self, iters):
		self.__test_stats = [self.TestStatistic(self.__RunModel()) for _ in range(iters)]


	def PValue(self, iters=1000) -> float:
		"""
		Calculate p-value.
		"""
		self.__permutate_test_statistics(iters)
		count = sum(1 for x in self.__test_stats if x >= self.actual)
		return count / iters

	def Confidence(self): raise NotImplementedError()
	def Power(self): raise NotImplementedError()


	def __MakeModel(self):
		"""
		Creates a pool from the two groups.
		"""
		group1, group2 = self.data
		self.n, self.m = len(group1), len(group2)
		self.pool = np.hstack((group1, group2))

	def __RunModel(self):
		"""
		Shuffle pool (randomisation).
		Create two groups (with length of group 1 and 2).
		"""
		np.random.shuffle(self.pool)
		data = self.pool[:self.n], self.pool[self.n:]
		return data

	def TestStatistic(self, *data): raise NotImplementedError() # 2 groups
