import numpy as np
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class PiecewiseLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 knots: ArrayLike=None,
                 name = None,
                 x_name = None):
        self.knots = np.array(knots) if knots is not None else None
        self.model = LinearRegression()
        self.name = name
        self.x_name = x_name

    def _transform(self, X: ArrayLike):
        X = np.asarray(X).reshape(-1, 1)
        if self.knots is None:
            return X
        
        # basis: x, max(0, x-k1), max(0, x-k2), ...
        pieces = [X]
        for k in self.knots:
            pieces.append(np.maximum(0, X - k))
        return np.hstack(pieces)

    def fit(self, X: ArrayLike, y: ArrayLike):
        X_new = self._transform(X)
        self.model.fit(X_new, y)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        X_new = self._transform(X)
        return self.model.predict(X_new)
    
class MultivariatePiecewiseLinear(BaseEstimator, RegressorMixin):
    """
    Piecewise linear model using hinge basis on each feature:
    
        φ = [x1, (x1-k11)+, (x1-k12)+, ..., x2, (x2-k21)+, ...]
    """
    def __init__(self, 
                 knots: ArrayLike=None,
                 name = None,
                 x_name = None):
        # knots must be a list: knots[i] = list of knots for feature i
        self.knots = knots
        self.model = LinearRegression()
        self.name = name
        self.x_name = x_name
        
    def _transform(self, X: ArrayLike):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        features = [X]  # always include raw linear terms

        if self.knots is not None:
            for feat_idx in range(n_features):
                if self.knots[feat_idx] is None:
                    continue
                for k in self.knots[feat_idx]:
                    h = np.maximum(0, X[:, feat_idx] - k).reshape(-1, 1)
                    features.append(h)

        return np.hstack(features)

    def fit(self, X: ArrayLike, y: ArrayLike):
        Z = self._transform(X)
        self.model.fit(Z, y)
        return self

    def predict(self, X: ArrayLike):
        Z = self._transform(X)
        return self.model.predict(Z)