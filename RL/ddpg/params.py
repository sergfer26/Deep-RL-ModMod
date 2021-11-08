import os
import sys
dir_path = os.path.dirname(os.path.realpath('../RL'))
dir_path+= '/RL'


sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))
from RL.params import PARAMS_ENV


PARAMS_DDPG = {'hidden_sizes': [64,64,64], 'actor_learning_rate': 1e-4, 'critic_learning_rate': 1e-3, 'gamma':0.98, 'tau':0.125, 'max_memory_size':int(1e5)}

PARAMS_UTILS = {'mu':0.0, 'theta': 0.00, 'max_sigma':0.01, 'min_sigma': 0.0, 'decay_period':int(PARAMS_ENV['TIME_MAX']*PARAMS_ENV['STEP']**-1)}

#El parametro decay_period debe ser menor o igual al numero de pasos por episodio
