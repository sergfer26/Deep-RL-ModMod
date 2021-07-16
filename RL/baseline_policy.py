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


def control_u4(x):
    '''Control de la temperatura del aire (T2)'''
    I = integral(x)
    try:
        error = x[-1] - x[-2]
    except:
        breakpoint()
    d = 0
    return control(error, KP4, KI4, KD4, I, d)

def control_u10(x):
    '''Control del CO2 (C1)'''
    I = integral(x)
    error = x[-1] - x[-2]
    if control_u10.old_error is None:
        control_u10.old_error = error
    d = error - control_u10.old_error
    control_u10.old_error = error
    return control(error, KP10, KI10, KD10, I, d)





class agent_baseline():
    def __init__(self, env):
        self.env = env
        self.reset()

    def get_action(self, state, T_out):
        U    = 0.5*np.ones(11)
        U[0] = self.u1(state,T_out) #U1
        self.T2_list += self.env.daily_T2[-5:]
        self.C1_list += self.env.daily_C1[-5:]
        U[3] = control_u4(self.T2_list[-5:]) #Ultimos 5 min        #U4
        U[9] = control_u10(self.C1_list[-5:]) #Ultimos 5 min        #U10
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
        self.T2_list = list()
        self.C1_list = list()
        control_u10.old_error = None
        