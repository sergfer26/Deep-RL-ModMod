import gym
import numpy as np
import pandas as pd
import numpy as np
import math
from time import time
from gym import spaces
import matplotlib.pyplot as plt
from progress.bar import Bar
from solver_climate import Climate_model
from solver_prod import GreenHouse
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
from get_indexes import Indexes

OUTPUTS = symbols('h nf') # variables de recompensa
CONTROLS = symbols('u3 u4 u7 u9 u10') # varibles de costo
R = 'min(0.01 * h, 10)'  # función de recompensa
P = '- (0.5/5) * (u3 + u4 + u7 + u9 + u10)' #función de penalización
symR = parse_expr(R)
symP = parse_expr(P)
reward_function = lambdify(OUTPUTS, symR)
penalty_function = lambdify(CONTROLS, symP)

LOW_OBS = np.zeros(6) # vars de estado de modelo clima + vars de estado de modelo prod (h, n)
HIGH_OBS = np.ones(6)
LOW_ACTION = np.zeros(10); # LOW_ACTION[7] = 0.5
HIGH_ACTION = np.ones(10)
STEP = 1/8 # día / # de pasos por día
TIME_MAX = 90 # días  
data_inputs = pd.read_csv('Inputs_Bleiswijk.csv')
INPUT_NAMES = list(data_inputs.columns)[0:-2]
SAMPLES = len(data_inputs) 
FRECUENCY = 60 # Frecuencia de medición de inputs del modelo del clima (minutos)
MONTH = '03' # Puede ser 'RANDOM'

class GreenhouseEnv(gym.Env):
    def __init__(self):
        self.dt = STEP # tamaño de paso de tiempo (días)
        self.frec = int(self.dt*24*60) # frecuencia de acciones de modelo climatico (minutos)
        self.action_space = spaces.Box(low=LOW_ACTION, high=HIGH_ACTION)
        self.observation_space = spaces.Box(low=LOW_OBS, high=HIGH_OBS)
        self.state_names = ['C1', 'RH', 'T', 'PAR', 'h', 'n']
        self.time_max = TIME_MAX
        self.dirClimate = Climate_model()
        self.dirGreenhouse = GreenHouse()
        self.i = 0
        self.indexes = Indexes(data_inputs,MONTH)
        self._reset()

    def is_done(self):
        if self.i == self.time_max/self.dt -1:
            return True
        else: 
            return False

    def get_reward(self, h, nf, action):
        _, _, u3, u4, _, _, u7, _, u9, u10 = action
        out = 0.0
        out += penalty_function(u3, u4, u7, u9, u10)
        out += reward_function(h, nf)
        return out

    def get_mean_data(self, data):
        end = int((self.i +1) * self.frec // FRECUENCY)
        start = int(end - 24 * 60/FRECUENCY) 
        mean = float(data[start: end].mean(skipna=True))
        return mean

    def step(self, action):
        if np.isnan(list(self.state.values())).any():
            breakpoint()
        self.dirClimate.update_controls(action)
        for minute in range(1, self.frec + 1):
            if minute % FRECUENCY == 0: # Los datos son de cada FRECUENCY minutos
                k = minute // FRECUENCY - 1
                self.update_vars_climate(k + self.i*self.frec/FRECUENCY) # 
            self.dirClimate.Run(Dt=1, n=1, sch=self.dirClimate.sch)
        
        reward = 0.0
        #old_h = self.dirGreenhouse.V('h')
        if (self.i + 1) % (1/self.dt) == 0: #Paso un dia
            C1M = self.dirClimate.OutVar('C1').mean() 
            TM = self.dirClimate.OutVar('T2').mean()
            PARM = self.get_mean_data(data_inputs['I2']) # PAR
            RHM = self.get_mean_data(data_inputs['RH'])
            self.dirGreenhouse.update_state(C1M, TM, PARM, RHM)
            self.dirGreenhouse.Run(Dt=1, n=1, sch=self.dirGreenhouse.sch)
            h = self.dirGreenhouse.V('h')
            n = self.dirGreenhouse.V('n')
            reward += reward_function(h, n)
        self.state = self.update_state()
        done = self.is_done()
        _, _, u3, u4, _, _, u7, _, u9, u10 = action
        reward += penalty_function(u3, u4, u7, u9, u10)
        self.i += 1
        state = np.array(list(self.state.values()))
        return state, reward, done
        
    def update_state(self):
        state = {k: self.dirGreenhouse.V(k) for k in self.state_names}
        return state
    
    def _reset(self):
        self.i = self.set_index()
        self.dirClimate.reset()
        self.dirGreenhouse.reset()
        T = self.dirClimate.V('T2')
        C1 = self.dirClimate.V('C1')
        PAR = float(data_inputs['I2'].iloc[self.i * (self.frec//FRECUENCY)])
        RH = float(data_inputs['RH'].iloc[self.i * (self.frec//FRECUENCY)])
        self.dirGreenhouse.update_state(C1, T, PAR, RH)
        self.state = self.update_state()
    
    def set_index(self):
        limit = ((SAMPLES -1) * FRECUENCY/(60) * 1/(24 * self.dt)) - self.time_max /self.dt
        if SEASON == 'RANDOM':
            return np.random.randint(0,limit)
        else:
            return np.random.choice(self.indexes)

    def reset(self):
        self._reset()
        state = np.array(list(self.state.values()))
        return state
    
    def n_random_actions(self, n):
        t1 = time()
        actions = np.random.uniform(0, 1,(n,10))
        h_vector = []
        bar = Bar('Processing', max=n)
        for action in actions:
            self.step(action)
            aux = np.array(list(self.state.values()))
            h_vector.append(aux[4])
            bar.next()
        bar.finish()
        t2 = time()
        plt.plot(range(n), h_vector)
        plt.suptitle('Incrementos de masa seca')
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

    def update_vars_climate(self, index):
        for name in INPUT_NAMES:
            self.dirClimate.Vars[name].val = data_inputs[name][index]

    def return_inputs_climate(self, start):
        end = start + self.time_max * 24 * (60//FRECUENCY)
        return data_inputs.iloc[start: end]



if __name__ == '__main__':
    env = GreenhouseEnv()
    env.n_random_actions(240) #30 días
