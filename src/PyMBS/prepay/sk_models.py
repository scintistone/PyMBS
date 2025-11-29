# A skealn DS approach to the problem

#1-Ds as ColumnTransformer, 2-D as individual cuestom transformer, 
# combined as Feature Union, then by simple multiplication/summation as estimator

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

class ToyTurnover(RegressorMixin, BaseEstimator):
    def __init__(self,         
                 TurnoverMax = 7.5,
                 TurnoverMaxAge = 30,
                 TurnoverMultiplier = 1.0):
        self.TurnoverMax = TurnoverMax
        self.TurnoverMaxAge = TurnoverMaxAge
        self.TurnoverMultiplier = TurnoverMultiplier

    def fit(self, X, y):
        return self

    def predict(self, X):
        '''X: wala'''
        return cpr2smm(self.TurnoverMultiplier * self.TurnoverMax * (2/ (1 + np.exp(-4.5 * X/self.TurnoverMaxAge))-1)/100)

class ToyRefinance(RegressorMixin, BaseEstimator):
    def __init__(self,         
                 RefiMax = 60,
                 RefiInflection = 150,
                 RefiRamp = 50,
                 RefiMultiplier = 1):
        self.RefiMax = RefiMax
        self.RefiInflection = RefiInflection
        self.RefiRamp = RefiRamp
        self.RefiMultiplier = RefiMultiplier

    def fit(self, X, y):
        return self

    def predict(self, X):
        '''X: incentive in bps'''
        return cpr2smm(self.RefiMultiplier * self.RefiMax / (1+np.exp(-(X-self.RefiInflection)/self.RefiRamp))/100)


class ToyModelRegressor(RegressorMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.sum(axis=1)
    
    def score(self, X, y, sample_weight = None):
        return super().score(X, y, sample_weight)