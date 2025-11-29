import numpy as np

from util import BAL, CPR2SMM, CDR2MDR

def SchCashFlow(coupon, orig_term, rem_term, balance):
    Month = np.arange(1, rem_term+1)
    RemTerm = rem_term - Month

    BegSchFactor = BAL(coupon, orig_term, RemTerm+1)
    EndSchFactor = BAL(coupon, orig_term, RemTerm)
    AmFactor = 1 - EndSchFactor/BegSchFactor

    BegSchBal = BegSchFactor*balance
    EndSchBal = EndSchFactor*balance
    SchPrin = AmFactor*balance
    SchInt = BegSchBal * coupon/1200

    results = {'Month': Month,
               'RemTerm': RemTerm,
               'BegSchFactor': BegSchFactor,
               'EndSchFactor': EndSchFactor,
               'AmFactor': AmFactor,
               'BegSchBal': BegSchBal,
               'EndSchBal': EndSchBal,
               'SchInt': SchInt,
               'SchPrin': SchPrin,
                }
    
    return results

def PoolCashFlow(coupon, orig_term, rem_term, balance, cpr = 0):
    '''Cash flow engine, only consider vol prepayment'''
    
    res = SchCashFlow(coupon, orig_term, rem_term, balance)

    Month = res['Month']
    RemTerm = res['RemTerm']

    BegSchFactor = res['BegSchFactor']
    EndSchFactor = res['EndSchFactor']
    AmFactor = res['AmFactor']

    BegSchBal = res['BegSchBal']
    EndSchBal = res['EndSchBal']
    SchPrin = res['SchPrin']
    SchInt = res['SchInt']

    smm = CPR2SMM(cpr)
    
    SMM = np.ones(rem_term) * smm
    CPR = np.ones(rem_term) * cpr                       

    SurvivalFactor = np.cumprod(1 - SMM/100)
    EndPoolFactor = SurvivalFactor * EndSchFactor
    BegPoolFactor = np.roll(EndPoolFactor, 1)
    BegPoolFactor[0] = 1

    BegActBal = BegPoolFactor * balance
    EndActBal = EndPoolFactor * balance
    ActInt = BegActBal * coupon/1200
    ActSchPrin = AmFactor * BegActBal
    PrePrin = (BegActBal - ActSchPrin) * SMM

    TotPrin = ActSchPrin + PrePrin

    CashFlow = TotPrin + ActInt

    results = {'Month': Month,
               'RemTerm': RemTerm,
               'BegSchFactor': BegSchFactor,
               'EndSchFactor': EndSchFactor,
               'AmFactor': AmFactor,
               'BegSchBal': BegSchBal,
               'EndSchBal': EndSchBal,
               'SchInt': SchInt,
               'SchPrin': SchPrin,
                'SMM': SMM,
                'CPR': CPR,
                'SurvivalFactor': SurvivalFactor,
                'BegPoolFactor': BegPoolFactor,
                'EndPoolFactor': EndPoolFactor,
                'BegActBal': BegActBal,
                'EndActBal': EndActBal,
                'ActInt': ActInt,
                'ActSchPrin': ActSchPrin,
                'PrePrin': PrePrin,
                'TotPrin': TotPrin,
                'CashFlow': CashFlow
                }
    
    return results


def PassthroughCashFlow(gross_coupon, net_coupon, orig_term, rem_term, bal, cpr = 0, smm = 0):
    res = PoolCashFlow(gross_coupon, orig_term, rem_term, bal, cpr, smm)
    PassthroughPrin = res['TotPrin']
    PassthroughInt = res['BegActBal'] * net_coupon/1200
    ServFee = res['BegActBal'] * (gross_coupon - net_coupon)/1200

    PassthroughCashFolow = PassthroughPrin + PassthroughInt

    res['PassthroughPrin'] = PassthroughPrin
    res['PassthroughInt'] = PassthroughInt
    res['PassthroughCashFolow'] = PassthroughCashFolow
    res['ServFee'] = ServFee
    return res

def MtgeCashFlowDefault(coupon, orig_term, rem_term, balance, cpr = 0, cdr = 0, severity=1, rec_lag = 12, advance = True):

    res = SchCashFlow(coupon, orig_term, rem_term, balance)

    Month = res['Month']
    RemTerm = res['RemTerm']

    BegSchFactor = res['BegSchFactor']
    EndSchFactor = res['EndSchFactor']
    AmFactor = res['AmFactor']

    BegSchBal = res['BegSchBal']
    EndSchBal = res['EndSchBal']
    SchPrin = res['SchPrin']
    SchInt = res['SchInt']

    smm = CPR2SMM(cpr)
    mdr = CDR2MDR(cdr)
    
    SMM = np.ones(rem_term) * smm
    CPR = np.ones(rem_term) * cpr
    MDR = np.ones(rem_term) * mdr
    CDR = np.ones(rem_term) * cdr

    NewDef = np.zeros(rem_term)
    ADB = np.zeros(rem_term)
    AmDef = np.zeros(rem_term)
    Fcl = np.zeros(rem_term)
    ExpAM = np.zeros(rem_term)
    ExpInt = np.zeros(rem_term)
    LostInt = np.zeros(rem_term)
    ActInt = np.zeros(rem_term)
    PrinLoss = np.zeros(rem_term)
    PrinRec = np.zeros(rem_term)
    VolPrepay = np.zeros(rem_term)
    ActAM = np.zeros(rem_term)
    PerfBal = np.zeros(rem_term)

    PrevPerfBal = balance
    PrevFcl = 0

    for i in Month:
        t = i-1

        AmFactor = (1 - EndSchFactor[t]/BegSchFactor[t])

        NewDef[t] = PrevPerfBal * MDR[t]
        if t >= rec_lag:
            if advance:
                ADB[t] = NewDef[t - rec_lag] * EndSchFactor[t]/BegSchFactor[t - rec_lag]
            else:
                ADB[t] = NewDef[t - rec_lag]
        else:
            ADB[t] = 0

        if advance:
            AmDef[t] = (NewDef[t] + PrevFcl - ADB[t]) *  AmFactor
        else:
            AmDef[t] = 0
        
        Fcl[t] = PrevFcl + NewDef[t] - ADB[t] - AmDef[t]

        ExpAM[t] = (PrevPerfBal + PrevFcl - ADB[t]) *  AmFactor
        ExpInt[t] = (PrevPerfBal + PrevFcl) * coupon/1200
        LostInt[t] = (NewDef[t] + PrevFcl) * coupon/1200
        ActInt[t] = ExpInt[t] - LostInt[t]

        if t >= rec_lag:
            PrinLoss[t] = np.min(NewDef[t-rec_lag] * severity, ADB[t])
        else:
            PrinLoss[t] = 0
        PrinRec[t] = np.max(ADB[t] - PrinLoss[t], 0)

        VolPrepay[t] = PrevPerfBal * EndSchFactor[t]/BegSchFactor[t] * SMM[t]
        ActAM[t] = (PrevPerfBal - NewDef[t]) *  AmFactor
        PerfBal[t] = PrevPerfBal - NewDef[t] - VolPrepay[t] - ActAM[t]

        PrevPerfBal = PerfBal[t]
        PrevFcl = Fcl[t]


    results = {'Month': Month,
               'RemTerm': RemTerm,
               'BegSchFactor': BegSchFactor,
               'EndSchFactor': EndSchFactor,
               'AmFactor': AmFactor,
               'BegSchBal': BegSchBal,
               'EndSchBal': EndSchBal,
               'SchInt': SchInt,
               'SchPrin': SchPrin,
                'SMM': SMM,
                'CPR': CPR,
                'NewDef': NewDef,
                'ADB': ADB,
                'AmDef': AmDef,
                'Fcl':Fcl,
                'ExpAM':ExpAM,
                'ExpInt': ExpInt ,
                'LostInt': LostInt ,
                'ActInt': ActInt,
                'PrinLoss': PrinLoss, 
                'PrinRec': PrinRec,
                'VolPrepay': VolPrepay, 
                'ActAM': ActAM, 
                'PerfBal': PerfBal 
                }
    return results