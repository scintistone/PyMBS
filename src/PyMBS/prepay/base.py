import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, RegularGridInterpolator

class Effect(object):
    def __init__(self,
                 X,
                 Y,
                 X_name,
                 Y_name,
                 custom_func = None):
        self._interp = custom_func
        self._X = X
        self._Y = Y
        self._Xname = X_name
        self._Yname = Y_name

    def __call__(self, x):
        return self._interp(x)

class Effect1D(Effect):
    def __init__(self,
                 X,
                 Y,
                 X_name,
                 Y_name,
                 custom_func = None):
        super().__init__(X,
                        Y,
                        X_name,
                        Y_name,
                        custom_func)
        if self._interp is None:
            self._interp = sp.interpolate.interp1d(X, 
                                                   Y, 
                                                   bounds_error=False, 
                                                   fill_value = (Y[0], Y[-1]))

class Effect2D(Effect):
    def __init__(self,
                 X,
                 Y,
                 X_name,
                 Y_name,
                 custom_func = None):
        super().__init__(X,
                        Y,
                        X_name,
                        Y_name,
                        custom_func)
        if self._interp is None:
            self._interp = sp.interpolate.RegularGridInterpolator(X, Y)


class Component(object):
    def __init__(self, name, effects = None, multiplicative = True):
        if effects is None:
            self._effects = []
        else:
            for e in effects:
                assert isinstance(e, Effect) or isinstance(e, Component), 'Effects must be instance of Effect or Component!'
            self._effects = effects
        self._name = name
        self._multiplicative = multiplicative

    def add_effect(self, effect):
        self._effects.append(effect)

    def __call__(self, x):
        if self._multiplicative:
            res = 1
        else:# additive
            res = 0
        
        self._cache = {}
        for e in self._effects:
            if isinstance(e, Effect1D):
                x_input = x[e._Xname]
            elif isinstance(e, Effect):
                x_input = [x[n] for n in e._Xname]
            elif isinstance(e, Component):
                x_input = x

            proj = e(x_input)
            self._cache[e._Yname] = proj
            
            if self._multiplicative:
                res = res * proj
            else:# additive
                res = res + proj

        self._cache['res'] = res
        return res