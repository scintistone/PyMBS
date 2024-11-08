import numpy as np

def smm2cpr(smm):
    return 1 - np.power(1 - smm, 12)


def cpr2smm(cpr):
    return 1 - np.power(1 - cpr, 1/12)

