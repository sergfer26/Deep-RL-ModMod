import sys
import gym
import pytz
import numpy as np
import pandas as pd
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates
from env import GreenhouseEnv, R, P, STEP, TIME_MAX, reward_function, penalty_function
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm
from math import ceil
from datetime import datetime, timezone
#from torch.utils.tensorboard import SummaryWriter
from get_report import create_report
from params import PARAMS_TRAIN

EPISODES = PARAMS_TRAIN['EPISODES']
STEPS = PARAMS_TRAIN['STEPS']
BATCH_SIZE = PARAMS_TRAIN['BATCH_SIZE']
SHOW = PARAMS_TRAIN['SHOW']
INDICE = PARAMS_TRAIN['INDICE'] #Cero para entrenar y 8770 para probar
tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute


env = GreenhouseEnv()
agent = DDPGagent(env)

noise = OUNoise(env.action_space)
action_dim =  env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
#writer_reward = SummaryWriter()
#writer_abs = SummaryWriter()
#writer_penalty = SummaryWriter()

def train_agent(agent, env, noise):
    rewards = []
    avg_rewards = []
    penalties = []
    abs_rewards = []
    for episode in range(EPISODES):
        #with tqdm(total=STEPS, position=0) as pbar:
        #pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
        state = env.reset()
        noise.reset()
        episode_reward = 0
        abs_reward = 0
        episode_penalty = 0
        for step in range(STEPS):
            action = agent.get_action(state)
            action = noise.get_action(action)
            new_state, reward, done = env.step(action) # modify
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)  
            _, _, u3, u4, _, _, u7, _, u9, u10 = action # modify
            p = -penalty_function(u3, u4, u7, u9, u10)
            r = 0.0
            #if (env.i + 1) % (1/env.dt) == 0:
            #    h = new_state[-2]; n = new_state[-1]
            episode_reward += reward
            abs_reward += reward - p
            episode_penalty += p
            #pbar.set_postfix(reward='{:.2f}'.format(episode_reward/STEPS), NF='{:2f}'.format(NF), H='{:2f}'.format(H))
            #pbar.update(1)      
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
    return rewards, avg_rewards, penalties, abs_rewards


###### Simulation ######
def sim(agent, env, indice = 0):
    state = env.reset() 
    start = env.i if indice == 0 else indice # primer indice de los datos
    env.i = start 
    #print('Voy a simular con indice = ', start)
    S_climate = np.zeros((STEPS, 4)) # vars del modelo climatico T1, T2, V1, C1
    S_data = np.zeros((STEPS, 2)) # datos recopilados RH PAR
    S_prod = np.zeros((STEPS, 6)) # datos de produccion h, nf, H, N, r_t, Cr_t
    A = np.zeros((STEPS, action_dim))
    episode_reward = 0.0
    for step in range(STEPS):
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
    data_inputs = env.return_inputs_climate(start)
    return S_climate, S_data, S_prod, A, data_inputs

def main():
    PATH = 'results_ddpg/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    mpl.style.use('seaborn')

    if len(sys.argv) == 1:
        pass
    else:
    # Load trained model 
        old_path = sys.argv[1:].pop()
        agent.load(old_path)
    
    rewards, avg_rewards, penalties, abs_rewards = train_agent(agent, env, noise)
    agent.save(PATH)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    fig.suptitle(r'$r_t =$ '+ R + r' $\cdot \mathbb{1}_{t = k days}$' + P + ', {} Days'.format(TIME_MAX), fontsize=14)

    ax1.plot(rewards, "-b", label='reward (DDPG)')
    ax1.plot(avg_rewards, "--b", label='avg reward (DDPG)', alpha=0.2)
    ax1.set_xlabel('episode')
    ax1.legend(loc='best')

    ax2.plot(abs_rewards, label='absolute reward', alpha=0.5)
    ax2.plot(penalties, label='penalty', alpha=0.5)
    ax2.set_xlabel('episode')
    ax2.legend(loc='best')

    if SHOW:
        plt.show()
    else:
        fig.savefig(PATH + '/reward.png')
        plt.close()


    
    S_climate, S_data, S_prod, A, data_inputs = sim(agent, env, indice = INDICE)

    df_climate = pd.DataFrame(S_climate, columns=('$T_1$', '$T_2$', '$V_1$', '$C_1$'))
    #df_climate['Date'] = dates
    #df_climate.set_index(['Date'], inplace=True)
    df_climate.plot(subplots=True, layout=(2, 2), figsize=(10, 7)) 
    #plt.tight_layout()
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_climate.png')
        plt.close()


    df_data = pd.DataFrame(S_data, columns=('RH','PAR'))
    df_data.plot(subplots=True, layout=(1, 2), figsize=(10, 7)) 
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_rh_par.png')
        plt.close()

    df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
    title='$H =$ {}, $NF=$ {}'.format(df_prod['$H$'].iloc[-1], df_prod['$N$'].iloc[-1])
    df_prod.plot(subplots=True, layout=(3, 2), figsize=(10, 7), title=title) 
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_prod.png')
        plt.close()

    dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$'))
    title= '$U$' # $U$
    dfa.plot(subplots=True, layout=(action_dim // 2, 2), figsize=(10, 7), title=title) 
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_actions.png')
        plt.close()


    data_inputs.set_index(['Date'], inplace=True) #Hace que el eje x sea la fecha
    data_inputs.plot(subplots=True, figsize=(10, 7))
    plt.tight_layout()
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_climate_inputs.png')
        plt.close()
    if not(SHOW):
        create_report(PATH)

if __name__=='__main__':
    main()