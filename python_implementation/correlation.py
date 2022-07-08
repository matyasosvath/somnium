

from hypothesistest import HypothesisTestPermute



class Correlation(HypothesisTestPermute):

    def __init__(self, data1, data2, assumption):
        self.data1 = data1
        self.data2 = data2
        self.__assumption_result = assumption(data1, data2) # missing
        




