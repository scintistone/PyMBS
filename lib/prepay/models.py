import numpy as np
from .base import Effect, Effect1D, Effect2D, Component
from ..util import cpr2smm, smm2cpr

# doesn't need this layer
# class PrepaymentModel(object):
#     def __init__(self, model_name):
#         self._model_name = model_name
#         self._model = []

#     def add_component(self, c):
#         self._model.append(c)

#     def __call__(self, x):
#         res = 0
#         for c in self._model:
#             res += c(x)
#         return res

class PSA(Component):
    def __init__(self, multiplier = 1):
        e = Effect1D([0, 30], [0, 0.06 * multiplier], 'wala', 'PSA')
        super().__init__('PSA', [e])
    def __call__(self, x):
        '''return smm'''
        return cpr2smm(super().__call__(x))


class RichardRoll(Component):
    '''CPR = RefiIncentive * SeasoningMultiplier * SeasonalityMultiplier * BurnoutMultiplier'''
    def __init__(self):
        
        RefiIncentive = Effect2D(None, None, 
                                 ('coupon', 'mrate'), 
                                 'RefiIncentive',
                                 lambda x: .2406 - .1389 * np.arctan(5.952 * (1.089 - x[0]/x[1])))

        Seasoning = Effect1D([0, 30], [0, 1], 'wala', 'Seasoning')

        #season_multiplier = np.asarray([.94, .76, .73, .96, .98, .92, .99, 1.1, 1.18, 1.21, 1.23, .97])
            
        Seasonality = Effect1D(None, None, 
                               'month', 
                               'Seasoning',
                               lambda m: np.asarray([.94, .76, .73, .96, .98, .92, .99, 1.1, 1.18, 1.21, 1.23, .97])[m-1])
        #Burnout: not implemented

        super().__init__('RichardRoll', [RefiIncentive, Seasoning, Seasonality])

    def __call__(self, x):
        '''return smm'''
        return cpr2smm(super().__call__(x))
    
class ToyModel(Component):
    def __init__(self):
        TurnoverMax = 7.5
        TurnoverMaxAge = 30
        TurnoverMultiplier = 1.0

        turnover = Effect1D(None, None, 
                            'wala', 'ht_cpr',
                            lambda x : TurnoverMultiplier * TurnoverMax * (2/ (1 + np.exp(-4.5 * x/TurnoverMaxAge))-1)/100)
        
        RefiMax = 60
        RefiInflection = 150
        RefiRamp = 50
        RefiMultiplier = 1

        refi = Effect1D(None, None,
                        'incentive', 'rf_cpr',
                        lambda x: RefiMultiplier * RefiMax / (1+np.exp(-(x-RefiInflection)/RefiRamp))/100)
        super().__init__('ToyModel', [turnover, refi], False)
    
    def __call__(self, x):
        '''return smm'''
        return cpr2smm(super().__call__(x))