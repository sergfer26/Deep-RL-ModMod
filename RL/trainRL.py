import sys
import gym
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from env import GreenhouseEnv, R, P, STEP, TIME_MAX, reward_function, penalty_function
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm
from math import ceil
from datetime import datetime
from datetime import datetime, timezone
import matplotlib as mpl
import pytz
#from torch.utils.tensorboard import SummaryWriter


EPISODES = 500
STEPS = int(TIME_MAX/STEP)
BATCH_SIZE = 32
SHOW = False

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute

PATH = 'results_ddpg/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
mpl.style.use('seaborn')

env = GreenhouseEnv()
agent = DDPGagent(env)
if len(sys.argv) == 1:
    pass
else:
    # Load trained model 
    old_path = sys.argv[1:].pop()
    agent.load(old_path)

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
            action = noise.get_action(action, step)
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
def sim(agent, env, noise):
    state = env.reset()
    S = np.zeros((STEPS, state_dim + 4))
    A = np.zeros((STEPS, action_dim))
    episode_reward = 0.0
    for step in range(STEPS):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)
        episode_reward += reward
        state = new_state
        S[step, 0:state_dim] = state
        S[step:, -4] = env.dirGreenhouse.V('H')
        S[step, -3] = env.dirGreenhouse.V('NF')
        S[step, -2] = reward
        S[step, -1] = episode_reward
        A[step, :] = action
        print(reward)
    return S, A


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

noise.max_sigma = 0.0
noise.min_sigma = 0.0
S, A = sim(agent, env, noise)

dfs = pd.DataFrame(S, columns=('$C_1$', '$RH$', '$T$', '$PAR$', '$h$', '$n$', '$H$', '$NF$', '$r_t$', '$Cr_t$'))
title='$H =$ {}, $NF=$ {}'.format(dfs['$H$'].iloc[-1], dfs['$NF$'].iloc[-1])
x = ceil((state_dim + 4)/2)
dfs.plot(subplots=True, layout=(x, 2), figsize=(10, 7), title=title) 
if SHOW:
    plt.show()
else:
    plt.savefig(PATH + '/sim_states.png')
    plt.close()

dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$'))
title= '$U$' # $U$
dfa.plot(subplots=True, layout=(action_dim // 2, 2), figsize=(10, 7), title=title) 
if SHOW:
    plt.show()
else:
    plt.savefig(PATH + '/sim_actions.png')
    plt.close()