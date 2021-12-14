from __future__ import barry_as_FLUFL
import sys
import numpy as np
import matplotlib as mpl
from env import GreenhouseEnv, STEP, TIME_MAX
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm
from correo import send_correo
import time
import os
# from torch.utils.tensorboard import SummaryWriter
# from typing_extensions import final
from get_report import create_report
from params import PARAMS_TRAIN,all_params, save_params
from get_report_constants import Constants
from graphics import save_Q, figure_reward, figure_state
from graphics import figure_rh_par, figure_prod, figure_actions 
from graphics import figure_inputs, compute_indexes, create_path
from graphics import save_rewards,figure_cost_gain
from benchmark import season1_nn,violin_reward,violin_actions
from sim import sim


EPISODES = PARAMS_TRAIN['EPISODES']
STEPS = PARAMS_TRAIN['STEPS']
BATCH_SIZE = PARAMS_TRAIN['BATCH_SIZE']
SHOW = PARAMS_TRAIN['SHOW']
SERVER = PARAMS_TRAIN['SERVER']
if SERVER: SHOW = False #No puedes mostrar nada en el servidor
INDICE = PARAMS_TRAIN['INDICE'] #Indice con el que se va a simular
SAVE_FREQ = PARAMS_TRAIN['SAVE_FREQ']


from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=SERVER)


def train_agent(agent, noise, path, episodes=EPISODES, save_freq=EPISODES):
    rewards = []
    avg_rewards = []
    penalties = []
    abs_rewards = []
    for episode in range(1, episodes + 1):
        with tqdm(total=STEPS, position=0) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
            env = GreenhouseEnv()
            state = np.array(list(env.state.values()))
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
            
        if episode % save_freq == 0:
            agent.save(path+'/nets', name="_{}".format(episode))

        rewards.append(episode_reward)
        abs_rewards.append(abs_reward)
        penalties.append(episode_penalty)
        avg_rewards.append(np.mean(rewards[-10:]))
        #writer_reward.add_scalar("Reward", episode_reward, episode)
        #writer_abs.add_scalar("Absolute reward", abs_reward, episode)
        #writer_penalty.add_scalar("Penalty", episode_penalty, episode)
    agent.save(path+'/nets')
    return rewards, avg_rewards, penalties, abs_rewards







def main():
    env = GreenhouseEnv()
    agent = DDPGagent(env)
    noise = OUNoise(env.action_space)
    action_dim =  env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    #writer_reward = SummaryWriter()
    #writer_abs = SummaryWriter()
    #writer_penalty = SummaryWriter()
    t1 = time.time()
    mpl.style.use('seaborn')
    if len(sys.argv) != 1:
    # Load trained model 
        PATH = sys.argv[1:].pop()
        print('Se cargo el modelo')
        agent.load(PATH + '/nets')
    else:
        PATH = create_path()

    Constants(PATH)
    save_params(all_params,PATH+'/output')
    #agent.actor.eval()
    #agent.critic.eval() 
    
    rewards, avg_rewards, penalties, abs_rewards = train_agent(agent, noise, PATH, save_freq=SAVE_FREQ)

    figure_reward(rewards, avg_rewards, penalties, abs_rewards,PATH)
    save_rewards(rewards, avg_rewards, penalties, abs_rewards,PATH)
    S_climate, S_data, S_prod, A, df_inputs,start = sim(agent, ind=INDICE)
    save_Q(env,PATH)
    figure_cost_gain(env,PATH)

    start = df_inputs['Date'].iloc[0]
    final_indexes = compute_indexes(start,STEPS,env.frec) #Es necesario crear nuevos indices para las graficas, depende de STEP
    figure_state(S_climate,final_indexes,PATH)
    figure_rh_par(S_data,final_indexes,PATH)
    figure_prod(S_prod,final_indexes,PATH)
    figure_actions(A,final_indexes,PATH)
    figure_inputs(df_inputs,PATH)
    
    t2 = time.time()
    season1_nn(agent,PATH,'')
    violin_reward(PATH,'nn') ##puede ser nn ó expert
    violin_actions(PATH,'nn')
    
    if not(SHOW):
        create_report(PATH,t2-t1)
        send_correo(PATH + '/reports/Reporte.pdf')
        pass
    

if __name__=='__main__':
    main()

