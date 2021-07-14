import gym
import numpy as np
import pandas as pd
import numpy as np
import math
from time import time
from gym import spaces
import matplotlib.pyplot as plt
#from progress.bar import Bar
from solver_climate import Climate_model
from solver_prod import GreenHouse
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
from get_indexes import Indexes
from params import PARAMS_ENV
#from reward import G, Qgas, Qco2

OUTPUTS = symbols('h nf') # variables de recompensa
CONTROLS = symbols('u3 u4 u7 u9 u10 C1') # varibles de costo y clima
R = PARAMS_ENV['R']   # función de recompensa
P = PARAMS_ENV['P'] # función de penalización
symR = parse_expr(R)
symP = parse_expr(P)
reward_function = lambdify(OUTPUTS, symR)
penalty_function = lambdify(CONTROLS, symP)

LOW_OBS = np.zeros(6) # vars de estado de modelo clima + vars de estado de modelo prod (h, n)
HIGH_OBS = np.ones(6)
LOW_ACTION = np.zeros(11); # LOW_ACTION[7] = 0.5
HIGH_ACTION = np.ones(11)
STEP = PARAMS_ENV['STEP']  # día / # de pasos por día
TIME_MAX = PARAMS_ENV['TIME_MAX'] # días  
data_inputs = pd.read_csv('Inputs_Bleiswijk.csv')
INPUT_NAMES = list(data_inputs.columns)[0:-2]
SAMPLES = len(data_inputs) 
FRECUENCY = PARAMS_ENV['FRECUENCY'] # Frecuencia de medición de inputs del modelo del clima (minutos)
SEASON = PARAMS_ENV['SEASON'] # Puede ser 'RANDOM'

class GreenhouseEnv(gym.Env):
    def __init__(self):
        self.dt = STEP # tamaño de paso de tiempo (días)
        self.frec = int(self.dt*24*60) # frecuencia de acciones de modelo climatico (minutos)
        self.action_space = spaces.Box(low=LOW_ACTION, high=HIGH_ACTION)
        self.observation_space = spaces.Box(low=LOW_OBS, high=HIGH_OBS)
        self.state_names = ['C1', 'RH', 'T', 'PAR', 'h', 'n']
        self.vars_cost = ['Qgas','Qco2','Qh2o']
        self.time_max = TIME_MAX
        self.limit = int(((SAMPLES -1) * FRECUENCY/(60) * 1/(24 * self.dt)) - self.time_max /self.dt) 
        self.dirClimate = Climate_model()
        self.dirGreenhouse = GreenHouse()
        self.i = 0
        self.indexes = Indexes(data_inputs[0:self.limit],SEASON) if SEASON != 'RANDOM' else None
        self.daily_C1  = list()
        self.daily_T2  = list()
        self.G_list    = list()
        self.Qvar_list = list()
        self._reset()


    def reset_daily_lists(self):
        '''Recrea las listas para promedios diarios '''
        self.daily_C1 = list([0])
        self.daily_T2 = list([0])

    def reset_cost(self,vars):
        '''Regresa las variables de costo a 0'''
        for name in vars:
            self.dirClimate.Vars[name].val = 0
    
    def reward_cost(self,vars):
        '''Resta los costos de produccion al reward'''
        reward = 0.0
        for name in vars:
            reward -= 60*self.dirClimate.OutVar(name) #De minutos a segundos 
        return reward

    def G(self,h):
        '''Precio de venta'''
        return 0.015341*h


    def is_done(self):
        if self.i == self.time_max/self.dt -1:
            return True
        else: 
            return False

    def get_mean_data(self, data):
        end = int((self.i +1) * self.frec // FRECUENCY)
        start = int(end - 24 * 60/FRECUENCY) 
        mean = float(data[start: end].mean(skipna=True))
        return mean

    def step(self, action):
        if np.isnan(list(self.state.values())).any():
            breakpoint()
        self.dirClimate.update_controls(action)
        C1 = list(); T = list()
        for minute in range(1, self.frec + 1):
            if minute % FRECUENCY == 0: # Los datos son de cada FRECUENCY minutos
                k = minute // FRECUENCY - 1
                self.update_vars_climate(k + self.i*self.frec//FRECUENCY) # 
            self.dirClimate.Run(Dt=1, n=1, sch=self.dirClimate.sch)
            C1.append(self.dirClimate.OutVar('C1'))
            T.append(self.dirClimate.OutVar('T2')) 
        reward = self.reward_cost(self.vars_cost)
        self.Qvar_list.append(float(reward))
        self.reset_cost(self.vars_cost)
        self.daily_C1 += C1
        self.daily_T2 += T
        if (self.i + 1) % (1/self.dt) == 0: #Paso un dia
            C1M = float(np.mean(self.daily_C1)) 
            TM = float(np.mean(self.daily_T2))
            self.reset_daily_lists()
            PARM = self.get_mean_data(data_inputs['I2']) # PAR
            RHM = self.get_mean_data(data_inputs['RH'])
            self.dirGreenhouse.update_state(C1M, TM, PARM, RHM)
            self.dirGreenhouse.Run(Dt=1, n=1, sch=self.dirGreenhouse.sch)
            h = self.dirGreenhouse.V('h')
            reward += self.G(h)
        self.state = self.update_state()
        done = self.is_done()
        self.i += 1
        state = np.array(list(self.state.values()))
        return state, reward, done
        
    def update_state(self):
        state = {k: self.dirGreenhouse.V(k) for k in self.state_names}
        state['T'] = self.dirClimate.V('T2')
        state['C1'] = self.dirClimate.V('C1')
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
        if SEASON == 'RANDOM':
            return np.random.RandomState().randint(0,self.limit)
        else:
            return np.random.RandomState().choice(self.indexes)

    def reset(self):
        self._reset()
        state = np.array(list(self.state.values()))
        return state
    
    def n_random_actions(self, n):
        t1 = time()
        actions = np.random.uniform(0, 1,(n,10))
        h_vector = []
        #bar = Bar('Processing', max=n)
        for action in actions:
            self.step(action)
            aux = np.array(list(self.state.values()))
            h_vector.append(aux[4])
            #bar.next()
        #bar.finish()
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

