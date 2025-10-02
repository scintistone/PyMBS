import numpy as np

def smm2cpr(smm):
    return 1 - np.power(1 - smm, 12)


def cpr2smm(cpr):
    return 1 - np.power(1 - cpr, 1/12)

def BAL(coupon, orig_term, rem_term):
    '''Scheduled amortized balance'''
    return (1 - np.power(1+coupon/1200, -rem_term))/(1 - np.power(1+coupon/1200, -orig_term))

def PMT(coupon, term):
    '''Monthly payment'''
    return coupon/1200/(1 - np.power(1+coupon/1200, -term))