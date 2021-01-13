import gym
import numpy as np
import pandas as pd
import numpy as np
import math
from time import time
from gym import spaces
import matplotlib.pyplot as plt
from progress.bar import Bar
from solver_climate import Dir as DirClimate
from solver_prod import Model as Greenhouse
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr

OUTPUTS = symbols('h nf')
CONTROLS = symbols('u1 u2 u3 u4 u5 u6 u7 u8 u9 u10')
R = 'h / nf' 
P = ' - 0.001 * (u1 + u2 + u3 + u5 + u6 + u7 + u8 + u9 + u10) - 10 * u4'
symR = parse_expr(R)
symP = parse_expr(P)
reward = lambdify(OUTPUTS, symR)
penalty = lambdify(CONTROLS, symP)
LOW_OBS = np.array([0, 0, 0, 0, 0, 0, 0, 0]) # vars de estado de modelo clima + vars de estado de modelo prod 
HIGH_OBS = np.array([1, 1, 1, 1, 1, 1, 1, 1])
TIME_MAX = 115
STEP = 1

data_par = pd.read_csv('PARout.csv')
data_rh = pd.read_csv('RHair.csv')
samples = data_par.shape[0] # 33133

class GreenhouseEnv(gym.Env):
    def __init__(self):
        self.dt = STEP # numéros días
        self.action_space = spaces.Box(low=0*np.ones(10), high=np.ones(10))
        self.observation_space = spaces.Box(low=LOW_OBS, high=HIGH_OBS)
        #PAR = c *max(sin, 0)
        self.state_names = ['C1', 'RH', 'T', 'PAR', 'H', 'NF', 'h', 'n']
        self.time_max = TIME_MAX
        self.dirClimate = DirClimate
        self.dirGreenhouse = Greenhouse
        self.state = self.reset()
        self.i = 0

    def is_done(self):
        if self.i == self.time_max/self.dt -1:
            return True
        else: 
            return False

    def get_reward(self, h, nf, action):
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = action
        out = 0.0
        out += penalty(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10)
        if nf:
            out += reward(h, nf)
            return out
        else:
            return out

    def get_mean_data(self, data):
        N = 12 * 24 # 12 saltos de 5 min en una hora
        k = self.i % samples
        mean = np.mean(data[k * N:(k+1) * N])
        if math.isnan(mean):
            return 0.0
        else:
            return float(mean)

    def step(self, action):
        self.dirClimate.Modules['Module1'].update_controls(action)
        self.dirClimate.Run(Dt=1, n=self.dt*24*60, sch=self.dirClimate.sch)
        C1M = self.dirClimate.OutVar('C1').mean()
        TM = self.dirClimate.OutVar('T2').mean()
        PARM = self.get_mean_data(data_par)
        RHM = self.get_mean_data(data_rh)
        self.dirGreenhouse.update_state(C1M, TM, PARM, RHM)
        self.dirGreenhouse.Run(Dt=1, n=1, sch=self.dirGreenhouse.sch)
        self.state = self.update_state()
        done = self.is_done()
        if done:
            reward = self.get_reward(self.state['NF'], self.state['H'], action)
        else:
            reward = 0
        self.i += 1
        return self.state, reward, done
        
        # 240 - Temperatura del aire
        # 244 - PAR
    def update_state(self):
        state = {k: self.dirGreenhouse.V(k) for k in self.state_names}
        return state
    
    def reset(self):
        self.i = 0
        self.dirClimate.Modules['Module1'].reset()
        self.dirGreenhouse.reset()
        T = np.random.normal(21, 2)
        C1 = np.random.normal(500, 1)
        PAR = float(data_par.iloc[0])
        RH = float(data_rh.iloc[0])
        self.dirGreenhouse.update_state(C1, T, PAR, RH)
        self.state = self.update_state()
        return self.update_state()
    
    
    def n_random_actions(self,n):
        t1 = time()
        actions = np.random.uniform(0,1,(n,10))
        H_vector = []
        bar = Bar('Processing', max=n)
        for action in actions:
            self.step(action)
            aux = np.array(list(self.state.values()))
            H_vector.append(aux[4])
            bar.next()
        bar.finish()
        t2 = time()
        plt.plot(range(n),H_vector)
        plt.suptitle('Masa seca')
        plt.show()
        self.imagen_accion(actions,n,t2-t1)

    def imagen_accion(self, A, n, tiempo):
        A = np.array(A)
        t = range(n)
        fig = plt.figure()
        for i in range(10):
            ax = fig.add_subplot(5,2, i+1)
            ax.plot(t, A[:,i])
            ax.set_title('$a_{}$'.format(i))
        plt.suptitle('Tiempo de ejecucion = ' + str(round(tiempo,2)))
        plt.show()