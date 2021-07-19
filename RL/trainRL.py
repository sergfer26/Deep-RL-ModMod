from __future__ import barry_as_FLUFL
import sys
from typing_extensions import final
import gym
import pytz
import numpy as np
import pandas as pd
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates
from env import GreenhouseEnv, STEP, TIME_MAX
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm
from math import ceil
from datetime import datetime, timezone
from time import time
from correo import send_correo
import json
from time import time
#from torch.utils.tensorboard import SummaryWriter
from get_report import create_report
from params import PARAMS_TRAIN
from get_report_constants import constants
from graphics import save_Q,figure_reward,figure_state,figure_rh_par,figure_prod,figure_actions,figure_inputs,compute_indexes,create_path, save_rewards,figure_cost_gain
EPISODES = PARAMS_TRAIN['EPISODES']
STEPS = PARAMS_TRAIN['STEPS']
BATCH_SIZE = PARAMS_TRAIN['BATCH_SIZE']
SHOW = PARAMS_TRAIN['SHOW']
INDICE = PARAMS_TRAIN['INDICE'] #Cero para entrenar y 8770 para probar

env = GreenhouseEnv()
agent = DDPGagent(env)

noise = OUNoise(env.action_space)
action_dim =  env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
#writer_reward = SummaryWriter()
#writer_abs = SummaryWriter()
#writer_penalty = SummaryWriter()

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train_agent(agent, env, noise,PATH):
    rewards = []
    avg_rewards = []
    penalties = []
    abs_rewards = []
    for episode in range(EPISODES):
        with tqdm(total=STEPS, position=0) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
            state = env.reset()
            noise.reset()
            episode_reward = 0
            abs_reward = 0
            episode_penalty = 0
            for step in range(STEPS):
                action                  = agent.get_action(state)
                action                  = noise.get_action(action)
                new_state, reward, done = env.step(action) # modify
                agent.memory.push(state, action, reward, new_state, done)
                if len(agent.memory) > BATCH_SIZE:
                    agent.update(BATCH_SIZE)  
                episode_reward          += float(reward)
                Ganancia                = float(env.G(env.state['h'])) if step % 24 == 0 else 0
                abs_reward              += Ganancia
                Costo                   = -float(reward) if step % 24 == 0 else -(reward - Ganancia)
                episode_penalty         += float(Costo)
                pbar.set_postfix(reward='{:.2f}'.format(episode_reward/STEPS), NF='{:2f}'.format(float(env.dirGreenhouse.V('NF'))), H='{:2f}'.format(float(env.dirGreenhouse.V('H'))))
                pbar.update(1)      
                state = new_state
                if done:
                    #sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                    break
        rewards.append(episode_reward)
        abs_rewards.append(abs_reward)
        penalties.append(episode_penalty)
        avg_rewards.append(np.mean(rewards[-10:]))
        #writer_reward.add_scalar("Reward", episode_reward, episode)
        #writer_abs.add_scalar("Absolute reward", abs_reward, episode)
        #writer_penalty.add_scalar("Penalty", episode_penalty, episode)
    agent.save(PATH)
    return rewards, avg_rewards, penalties, abs_rewards


###### Simulation ######
#from progressbar import*

def sim(agent, env, indice = 0):
    #pbar = ProgressBar(maxval=STEPS)
    #pbar.start()
    state = env.reset() 
    start = env.i if indice == 0 else indice # primer indice de los datos
    env.i = start 
    #env.i = 75992 
    #print('Se simula con indice = ', env.i)
    S_climate = np.zeros((STEPS, 4)) # vars del modelo climatico T1, T2, V1, C1
    S_data = np.zeros((STEPS, 2)) # datos recopilados RH PAR
    S_prod = np.zeros((STEPS, 6)) # datos de produccion h, nf, H, N, r_t, Cr_t
    A = np.zeros((STEPS, action_dim))
    episode_reward = 0.0
    for step in range(STEPS):
        #pbar.update(step)
        action = agent.get_action(state)
        new_state, reward, done = env.step(action)
        episode_reward += reward
        C1, RH, T2, PAR, h, n = state
        T1 = env.dirClimate.V('T1'); V1 = env.dirClimate.V('V1')
        S_climate[step, :] = np.array([T1, T2, V1, C1]) 
        H = env.dirGreenhouse.V('H'); NF= env.dirGreenhouse.V('NF')
        S_data[step, :] = np.array([RH, PAR])
        S_prod[step, :] = np.array([h, n, H, NF, reward, episode_reward])
        A[step, :] = action
        state = new_state
    #pbar.finish()
    data_inputs = env.return_inputs_climate(start)
    return S_climate, S_data, S_prod, A, data_inputs,start



def main():

    t1 = time()
    mpl.style.use('seaborn')
    PATH = create_path()
    constants(PATH)
    if len(sys.argv) != 1:
    # Load trained model 
        old_path = sys.argv[1:].pop()
        print('Se cargo el modelo')
        agent.load(old_path)
    '''
    rewards, avg_rewards, penalties, abs_rewards = train_agent(agent, env, noise,PATH)

    figure_reward(rewards, avg_rewards, penalties, abs_rewards,PATH)
    save_rewards(rewards, avg_rewards, penalties, abs_rewards,PATH)

    S_climate, S_data, S_prod, A, df_inputs,start = sim(agent, env, indice = INDICE)
    save_Q(env,PATH)
    figure_cost_gain(env,PATH)
    
    
    final_indexes = compute_indexes(start,STEP,TIME_MAX) #Es necesario crear nuevos indices para las graficas, depende de STEP
    figure_state(S_climate,final_indexes,PATH)
    figure_rh_par(S_data,final_indexes,PATH)
    figure_prod(S_prod,final_indexes,PATH)
    figure_actions(A,final_indexes,action_dim,PATH)
    figure_inputs(df_inputs,final_indexes,PATH)
    
    t2 = time()
    if not(SHOW):
        create_report(PATH,t2-t1)
        send_correo(PATH + '/Reporte.pdf'
    '''

if __name__=='__main__':
    main()
