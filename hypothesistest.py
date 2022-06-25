class HypothesisTestPermute(object):

	def __init__(self, data):
		self.n : int 
		self.m : int
		self.data : tuple(x,y)
		self.pool: list	
		self.test_stats # private List<double>

		self.MakeModel()
		self.actual = self.TestStatistic(data)

	def __permutate_test_statistics(self, iters):
		self.test_stats = [self.TestStatistic(self.RunModel()) for _ in range(iters)]


	def PValue(self,iters=1000):
		self.__permutate_test_statistics(iters)
		count = sum(1 for x in self.test_stats if x >= self.actual)
		return count / iters

	def Confidence(self): raise UnimplementedMethodException()
	def Power(self): raise UnimplementedMethodException()


	def MakeModel(self):
		"""
		Creates a pool from the two groups.
		"""
		group1, group2 = self.data
		self.n, self.m = len(group1), len(group2)
		self.pool = np.hstack((group1, group2))

	def RunModel(self):
		"""
		Shuffle pool (randomisation).
		Create two group (with length of group 1 and 2).
		"""
		np.random.shuffle(self.pool)
		data = self.pool[:self.n], self.pool[self.n:]
		return data

	def TestStatistic(self, x,y): raise UnimplementedMethodException() # 2 groups
	def TestStatistic(self, x,y,z): raise UnimplementedMethodException() # 3 groups
	def TestStatistic(self, data): raise UnimplementedMethodException() # more than 3 groups



class DiffMeansPermute(thinkstats2.HypothesisTest):


	def TestStatistic(self, data):
		group1, group2 = data
		test_stat = abs(group1.mean() - group2.mean())
		return test_stat
	
