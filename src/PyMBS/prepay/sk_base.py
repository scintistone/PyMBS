import numpy as np
from numpy.typing import ArrayLike

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

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
                 x_name:list = None,
                 flat_extrapolation: bool = True):
        # knots must be a list: knots[i] = list of knots for feature i
        self.knots = knots
        self.model = LinearRegression()
        self.name = name
        self.x_name = x_name
        self.flat_extrapolation = flat_extrapolation
        
    def _flat_extrapolate(self, X):
        """Clamp each feature into the knot range for flat extrapolation."""
        X = np.asarray(X).copy()
        if self.knots is None:
            return X
        
        # Clamp X[:, i] to [min_knot_i, max_knot_i]
        for feat_idx, feat_knots in enumerate(self.knots):
            if feat_knots is None or len(feat_knots) == 0:
                continue
            
            kmin = min(feat_knots)
            kmax = max(feat_knots)
            X[:, feat_idx] = np.clip(X[:, feat_idx], kmin, kmax)
        
        return X

    def _transform(self, X):
        """Create hinge basis after flat extrapolation."""
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # First apply flat extrapolation
        if self.flat_extrapolation:
            Xe = self._flat_extrapolate(X)
        else:
            Xe = X
        
        # Start with linear terms
        features = [Xe]
        
        # Add hinge features
        if self.knots is not None:
            for feat_idx in range(n_features):
                if self.knots[feat_idx] is None:
                    continue
                for k in self.knots[feat_idx]:
                    h = np.maximum(0, Xe[:, feat_idx] - k).reshape(-1, 1)
                    features.append(h)
        
        return np.hstack(features)

    def fit(self, X: ArrayLike, y: ArrayLike):
        Z = self._transform(X)
        self.model.fit(Z, y)
        return self

    def predict(self, X: ArrayLike):
        Z = self._transform(X)
        return self.model.predict(Z)
    
class AdditivePiecewiseLinear(BaseEstimator, RegressorMixin):
    """
    A model that sums multiple MultivariatePiecewiseLinear estimators.

    Parameters
    ----------
    components : list
        List of MultivariatePiecewiseLinear instances.

    train_mask : list of bool
        Same length as components. True → this component is fitted, False → frozen.
    """

    def __init__(self, components:list, train_mask=None):
        self.components = components
        if train_mask is None:
            self.train_mask = [True] * len(components)
        else:
            assert(len(train_mask) == len(components), "train_mask must be same length as components")

    # ---- param vector operations ----

    def _split_vector(self, full_vec):
        """
        Split full_vec into sub-vectors for each component.
        """
        sizes = [len(c.get_param_vector()) for c in self.components]
        out = []
        idx = 0
        for s in sizes:
            out.append(full_vec[idx:idx+s])
            idx += s
        return out

    def _full_vector(self):
        """
        Assemble full parameter vector (trainable + frozen).
        """
        return np.concatenate([c.get_param_vector() for c in self.components])
    
    def set_train_mask(self, train_mask:list):
        """
        Mask to extract only trainable parameters.
        """
        assert(len(train_mask) == len(self.components), "train_mask must be same length as components")
        self.train_mask = train_mask
        # masks = []
        # for c in self.components:
        #     n = len(c.get_param_vector())
        #     if c.trainable:
        #         masks.append(np.ones(n, dtype=bool))
        #     else:
        #         masks.append(np.zeros(n, dtype=bool))
        # return np.concatenate(masks)

    def _vector_from_trainable(self, trainable_v):
        """
        Insert optimized trainable parameters back into full vector.
        """
        full = self._full_vector().copy()
        mask = self.self.train_mask
        full[mask] = trainable_v
        return full

    def _set_full_vector(self, full_vec):
        """
        Push parameters into each component.
        """
        subvectors = self._split_vector(full_vec)
        for sv, c in zip(subvectors, self.components):
            c.set_param_vector(sv)

    def predict(self, X):
        preds = [c.predict(X[c.x_name]) for c in self.components]
        return np.sum(preds, axis=0)

    def fit(self, X, y, sample_weight = None):
        """
        Joint fitting: optimize all trainable components simultaneously.
        """

        def loss_fn(trainable_vec):
            full_vec = self._vector_from_trainable(trainable_vec)
            self._set_full_vector(full_vec)
            pred = self.predict(X)
            return root_mean_squared_error(y, pred, sample_weight) #np.mean((pred - y)**2)

        init_full = self._full_vector()
        mask = self.train_mask
        init_trainable = init_full[mask]

        res = minimize(loss_fn, init_trainable, method='L-BFGS-B')

        # Update model with optimized params
        best_full_vec = self._vector_from_trainable(res.x)
        self._set_full_vector(best_full_vec)

        return self