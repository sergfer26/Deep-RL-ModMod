import numpy as np
import pandas as pd
from climate_model.functions import q2
from scipy.interpolate import interp1d
from params_control import PARAMS_CONTROL
from params import minutos


data      = pd.read_csv('Inputs_Bleiswijk.csv')
KP4       = PARAMS_CONTROL['KP4']
KI4       = PARAMS_CONTROL['KI4']
KD4       = PARAMS_CONTROL['KD4']

KP10      = PARAMS_CONTROL['KP10']
KI10      = PARAMS_CONTROL['KI10']
KD10      = PARAMS_CONTROL['KD10']


VENTSET1 = 25
VENTSET2 = 25

NIGHT_PAR = 5

KP8       = PARAMS_CONTROL['KP8']
KI8       = PARAMS_CONTROL['KI8']
KD8       = PARAMS_CONTROL['KD8']

VENTSET1  = PARAMS_CONTROL['VENTSET1']
VENTSET2  = PARAMS_CONTROL['VENTSET2']

NIGHT_PAR = PARAMS_CONTROL['NIGHT_PAR']
OTHER_PAR = PARAMS_CONTROL['OTHER_PAR']

def set_point_t2(self, par):
    if par[-1] < PAR_NIGHT:
        self.heatset = 16
    else:
        self.heatset = 20
            
def set_point_co2(self,state):
        if state[3] > 5:
            self.co2 = 900
        else:
            self.co2 = 400

PBAND1    = PARAMS_CONTROL['PBAND1']
PBAND2    = PARAMS_CONTROL['PBAND2']

T_OUT1    = PARAMS_CONTROL['T_OUT1']
T_OUT2    = PARAMS_CONTROL['T_OUT2']

VPD_SET1  = PARAMS_CONTROL['VPD_SET1']
VPD_SET2  = PARAMS_CONTROL['VPD_SET2']

HEATSET1  = PARAMS_CONTROL['HEATSET1']
HEATSET2  = PARAMS_CONTROL['HEATSET2']

CO2SET1   = PARAMS_CONTROL['CO2SET1']
CO2SET2   = PARAMS_CONTROL['CO2SET2']

V         = PARAMS_CONTROL['V']


def set_varset(par):
    vpd_set  = VPD_SET1 if par < NIGHT_PAR else VPD_SET2
    vent_set = VENTSET1 if par < NIGHT_PAR else VENTSET2
    heat_set = HEATSET1 if par < NIGHT_PAR else HEATSET2
    co2_set  = CO2SET1  if par < NIGHT_PAR else CO2SET2
    return vpd_set, vent_set, heat_set, co2_set

            
def set_pband(T_out, Vel):
    Pband = 0
    if T_out < T_OUT1:
        Pband = PBAND1
    elif T_out > T_OUT2:
        Pband =  PBAND2
    else:
        m = (PBAND2 - PBAND1) / (T_OUT2 - T_OUT1)
        Pband = PBAND2 - m * (T_out - T_OUT1)
    
    Pband_corr = 0.5 * (Vel - V) if Vel >= V else 0
    return Pband - Pband_corr


def pid(error, kp, ki, kd, I, d):
    return kp * error + ki * I + kd * d


def integral(X):
    a = X[0]
    b = X[-1]
    return np.mean(X)*(b-a)
    
def control(error, kp, ki, kd, I, d):
    PID = pid(error, kp, ki, kd, I, d)
    return 0.001 * max(min(PID, 100), 0)


def control_u1(par, T_out):
    '''Control de la pantalla termica'''
    if NIGHT_PAR < par < OTHER_PAR and T_out < 10:
        return 0.5
    elif par < NIGHT_PAR:
        return 1
    else: 
        return 0


def control_u4(x,set_):
    '''Control de la temperatura del aire (T2)'''
    I = integral(x-set_)
    try:
        error = x[-1] - set_
    except:
        breakpoint()
    d = 0
    return control(error, KP4, KI4, KD4, I, d)


def control_u10(x, set_):
    '''Control del CO2 (C1)'''
    I = integral(x - set_)
    error = x[-1] - set_
    if control_u10.old_error is None:
        control_u10.old_error = error
    d = error - control_u10.old_error
    control_u10.old_error = error
    return control(error, KP10, KI10, KD10, I, d)


def control_u9(T_out, vent_set, Pband ):
    '''Control del sistema de niebla'''
    if T_out < vent_set:
        return 0.0
    elif T_out > vent_set + Pband:
        return 1.0
    else:
        return (T_out - vent_set)/Pband


def control_u8(vpd, vpd_set):
    '''Control de los respiraderos en el techo'''
    I = integral(vpd - vpd_set)
    error = vpd[-1] - vpd_set
    if control_u8.old_error is None:
        control_u8.old_error = error
    d = error - control_u8.old_error
    control_u8.old_error = error
    control = pid(error, KP8, KI8, KD8, I, d)
    U_min = 0.02 if vpd[-1] < 0.5 else 0
    return min(max(0, control + U_min), 1) 


class agent_baseline():
    def __init__(self):
        self.i       = 0
        self.reset()

    def get_action(self, i, env):
        par   = self.get_straight_line('I2',i)
        T_out = self.get_straight_line('I5',i)
        Vel   = self.get_straight_line('I8',i)
        U    = np.zeros(11)
        T1_list = np.array([float(y) for y in env.daily_T1[-5:]])
        T2_list = np.array([float(y) for y in env.daily_T2[-5:]])
        C1_list = np.array([float(y) for y in env.daily_C1[-5:]])
        V1_list = np.array([float(y) for y in env.daily_V1[-5:]])
        pband = set_pband(T_out[-1], Vel[-1])

        vpd_set, vent_set, heat_set, co2_set = set_varset(par[-1])
        VPD = np.array(list(map(lambda t1,v1:q2(t1) - v1, T1_list, V1_list)))
        U[0] = control_u1(par[-1], T_out[-1]) #U1
        U[3] = control_u4(T2_list, heat_set) #Ultimos 5 min        U4
        U[7] = control_u8(VPD, vpd_set)
        U[8] = control_u9(T_out[-1], vent_set, pband)
        U[9] = control_u10(C1_list, co2_set)     #Ultimos 5 min        U10
        return U

    def get_straight_line(self, key, k):
        y1 = data[key][k]
        y2 = data[key][k + 1]
        y = interp1d([0, 1], [y1, y2], kind='linear')
        x = np.linspace(0, 1, 60)
        start = (self.i*minutos)%60 
        end = -1 if ((self.i+ 1) * (minutos))%60 == 0 else ((self.i+ 1) * (minutos))%60
        x = x[start:end]
        if len(x)== 0:
            breakpoint()
        self.i += 1
        return np.array(list(map(lambda t: float(y(t)), x)))
            
    def set_point_co2(self,state):
        if state[3] > 5:
            self.co2 = 900
        else:
            self.co2 = 400
    

    def reset(self):
        control_u10.old_error = None
        control_u8.old_error = None
        
