import numpy as np

KP4 = 3.672
KI4 = 188.659
KD4 = 0.00
KP10 = 0.019
KI10 = 6033.599
KD10 = 65.103


def pid(error, kp, ki, kd, I, d):
    return kp * error + ki * I + kd * d


def integral(X):
    a = X[0]
    b = X[-1]
    return np.mean(X)*(b-a)
    
def control(error, kp, ki, kd, I, d):
    PID = pid(error, kp, ki, kd, I, d)
    return 0.001 * max(min(PID, 100), 0)


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




class agent_baseline():
    def __init__(self, env):
        self.env     = env
        self.heatset = 0
        self.co2     = 0
        self.reset()

    def get_action(self, state, T_out):
        U    = np.zeros(11)
        U[0] = self.u1(state,T_out) #U1
        T2_list = np.array([float(y) for y in self.env.daily_T2[-5:]])
        C1_list = np.array([float(y) for y in self.env.daily_C1[-5:]])
        self.set_point_t2(state)
        self.set_point_co2(state)
        U[10] = control_u4(T2_list,self.heatset) #Ultimos 5 min        U4
        U[9] = control_u10(C1_list,self.co2)    #Ultimos 5 min        U10
        self.reset()
        return U

    def set_point_t2(self,state):
        if state[3] < 5:
            self.heatset = 16
        else:
            self.heatset = 20
            
    def set_point_co2(self,state):
        if state[3] > 5:
            self.co2 = 900
        else:
            self.co2 = 400
    
    def u1(self,state,T_out):
        '''Control de la pantalla termica'''
        if 5 < state[3] < 50 and T_out < 10:
            return 0.5
        elif state[3] < 5:
            return 1
        else: 
            return 0

    def reset(self):
        control_u10.old_error = None
        