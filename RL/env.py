import gym
import pandas as pd
import numpy as np
from gym import spaces
from Model_Climate.climate_env import ClimateU
from Infer_V1.Solver_Pdn import Solver_Pdn

LOW_OBS = np.array([0, 0, 0, 0, 0]) # vars de estado de modelo clima + vars de estado de modelo prod 
HIGH_OBS = np.array([1, 1, 1, 1, 1])
TIME_MAX = 90
STEP = 1

class Greenhouse(gym.Env):
    def __init__(self):
        self.dt = STEP # numéros días
        self.action_space = spaces.Box(low = 0*np.ones(10), high = np.ones(10))
        self.observation_space = spaces.Box(low = LOW_OBS, high = HIGH_OBS)
        self.time_max = TIME_MAX
        self.state = self.reset()
        self.i = 0

    def is_done(self):
        if self.i == self.time_max/self.dt -1:
            return True
        else: 
            return False

    def get_reward(self, nf, h):
        return h/nf

    def step(self, action):
        C1M, V1M, T1M, T2M = ClimateU(action, self.dt)
        nf, h, T, PAR, C1, RH = Solver_Pdn(C1M, V1M, T1M, T2M, self.dt)
        self.state = np.array([C1M, V1M, T1M, T2M, PAR, RH])
        reward = self.get_reward(nf, h)
        done = self.is_done()
        self.i += 1
        return self.state, reward, done
        
        # 240 - Temperatura del aire
        # 244 - PAR
    
    def reset(self):
        self.i = 0
        self.state = np.array([0, 0, 0, 0, 0]) # si podemos partir del mismo
        return self.state
