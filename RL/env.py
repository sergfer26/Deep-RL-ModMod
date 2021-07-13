import gym
import numpy as np
import pandas as pd
import numpy as np
import math
from time import time
from gym import spaces
import matplotlib.pyplot as plt
#from progress.bar import Bar
from climate_model.model import Climate_model
from solver_prod import GreenHouse
from get_indexes import Indexes
from params import PARAMS_ENV


LOW_OBS     = np.zeros(6) # vars de estado de modelo clima + vars de estado de modelo prod (h, n)
HIGH_OBS    = np.ones(6)
LOW_ACTION  = np.zeros(11) # LOW_ACTION[7] = 0.5
HIGH_ACTION = np.ones(11)
STEP        = PARAMS_ENV['STEP']  # día / # de pasos por día
TIME_MAX    = PARAMS_ENV['TIME_MAX'] # días  
data_inputs = pd.read_csv('Inputs_Bleiswijk.csv')
INPUT_NAMES = list(data_inputs.columns)[0:-2]
SAMPLES     = len(data_inputs) 
FRECUENCY   = PARAMS_ENV['FRECUENCY'] # Frecuencia de medición de inputs del modelo del clima (minutos)
MONTH       = PARAMS_ENV['MONTH'] # Puede ser 'RANDOM'


def G(H):
    ''' Ganancia en pesos por el peso H '''
    return 0.015341*H


class GreenhouseEnv(gym.Env):
    def __init__(self):
        self.dt = STEP # tamaño de paso de tiempo (días)
        self.frec = int(self.dt*24*60) # frecuencia de acciones de modelo climatico (minutos)
        self.action_space = spaces.Box(low=LOW_ACTION, high=HIGH_ACTION)
        self.observation_space = spaces.Box(low=LOW_OBS, high=HIGH_OBS)
        self.state_names = ['C1', 'RH', 'T', 'PAR', 'h', 'n']
        self.time_max = TIME_MAX
        self.limit = int(((SAMPLES -1) * FRECUENCY/(60) * 1/(24 * self.dt)) - self.time_max /self.dt) 
        self.dirClimate = Climate_model()
        self.dirGreenhouse = GreenHouse()
        self.i = 0
        self.indexes = Indexes(data_inputs[0:self.limit],MONTH) if MONTH != 'RANDOM' else None
        self.daily_C1 = list()
        self.daily_T2 = list()
        self._reset()

    def reset_daily_lists(self):
        '''Recrea las listas para promedios diarios '''
        self.daily_C1 = list()
        self.daily_T2 = list()

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
        Q_var = 0
        for minute in range(1, self.frec + 1):
            if minute % FRECUENCY == 0: # Los datos son de cada FRECUENCY minutos
                k = minute // FRECUENCY - 1
                self.update_vars_climate(k + self.i*self.frec//FRECUENCY) # 
            self.dirClimate.Run(Dt=1, n=1, sch=self.dirClimate.sch)
            C1.append(self.dirClimate.OutVar('C1'))
            T.append(self.dirClimate.OutVar('T2'))
            Q_var += self.dirClimate.OutVar('Qgas') 
            Q_var += self.dirClimate.OutVar('Qco2')
        
        reward = 0.0 
        Qh = Q_var - self.old_Qvar # Qvar de este paso
        reward -= Qh
        self.old_Qvar = Qh # Actualizo Qvar del paso anterior

        self.daily_C1 += C1
        self.daily_T2 += T
        #old_h = self.dirGreenhouse.V('h')
        if (self.i + 1) % (1/self.dt) == 0: #Paso un dia
            C1M = float(np.mean(self.daily_C1)) 
            TM = float(np.mean(self.daily_T2))
            self.reset_daily_lists()
            PARM = self.get_mean_data(data_inputs['I2']) # PAR
            RHM = self.get_mean_data(data_inputs['RH'])
            self.dirGreenhouse.update_state(C1M, TM, PARM, RHM)
            self.dirGreenhouse.Run(Dt=1, n=1, sch=self.dirGreenhouse.sch)
            h = self.dirGreenhouse.V('h')
            reward += G(h)
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
        self.old_Qvar = 0
        self.Qvar = 0
    
    def set_index(self):
        if MONTH == 'RANDOM':
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

