import gym
import numpy as np
import pandas as pd
import numpy as np
import math
from time import time
from gym import spaces
from progress.bar import Bar
from solver_climate import Dir as DirClimate
from solver_prod import Model as Greenhouse
import matplotlib.pyplot as plt

LOW_OBS = np.array([0, 0, 0, 0, 0, 0, 0, 0]) # vars de estado de modelo clima + vars de estado de modelo prod 
HIGH_OBS = np.array([1, 1, 1, 1, 1, 1, 1, 1])
TIME_MAX = 90
STEP = 1

data_par = pd.read_csv('PARout.csv')
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

    def get_reward(self, nf, h):
        if nf:
            return h/nf
        else:
            return 0

    def get_par(self):
        N = 12 * 24 # 12 saltos de 5 min en una hora
        k = self.i % samples
        par_mean = np.mean(data_par[k * N:(k+1) * N])
        if math.isnan(par_mean):
            return 0.0
        else:
            return float(par_mean)

    def step(self, action):
        self.dirClimate.Modules['Module1'].update_controls(action)
        self.dirClimate.Run(Dt=1, n=self.dt*24*60, sch=self.dirClimate.sch)
        C1M = self.dirClimate.OutVar('C1').mean()
        TM = self.dirClimate.OutVar('T2').mean()
        PARM = self.get_par()
        self.dirGreenhouse.update_state(C1M, TM, PARM)
        self.dirGreenhouse.Run(Dt=1, n=1, sch=self.dirGreenhouse.sch)
        self.state = self.update_state()
        reward = self.get_reward(self.state['NF'], self.state['H'])
        done = self.is_done()
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
        C1 = np.random.normal(3, 2)
        PAR = float(data_par.iloc[0])
        self.dirGreenhouse.update_state(C1, T, PAR)
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