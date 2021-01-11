import sys
import gym
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from env import GreenhouseEnv
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm
from datetime import datetime
from datetime import datetime, timezone
import pytz

EPISODES = 2
STEPS = 15
BATCH_SIZE = 64
SHOW = False

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute

PATH = 'results/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

#env = gym.make("Pendulum-v0")

def train_agent(agent, env, noise, steps=STEPS, episodes=EPISODES, batch_size=BATCH_SIZE):
    rewards = []
    avg_rewards = [] 
    for episode in range(EPISODES):
        #with tqdm(total=STEPS, position=0) as pbar:
        #pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
        state = env.reset()
        state = np.array(list(state.values()))
        noise.reset()
        episode_reward = 0
        for step in range(STEPS):
            action = agent.get_action(state)
            action = noise.get_action(action, step)
            new_state, reward, done = env.step(action) 
            new_state = np.array(list(new_state.values()))
            agent.memory.push(state, action, reward, new_state, done)
    
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)  
            #['C1', 'RH', 'T', 'PAR', 'H', 'NF', 'h', 'n']
            _, _, _, _, H, NF, _, _ = new_state
            episode_reward += reward
            #pbar.set_postfix(reward='{:.2f}'.format(episode_reward/STEPS), NF='{:2f}'.format(NF), H='{:2f}'.format(H))
            #pbar.update(1)      
    
            state = new_state
            if done:
                #sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                break
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    return rewards, avg_rewards

env = GreenhouseEnv()
agent = DDPGagent(env)
noise = OUNoise(env.action_space)


rewards, avg_rewards = train_agent(agent, env, noise)
agent.save(PATH)

plt.plot(rewards, "--b", label='reward (DDPG)', alpha=0.1)
plt.plot(avg_rewards, "-b", label='avg reward (DDPG)')
plt.legend(loc="lower right")

plt.xlabel('Episode')
plt.ylabel('Reward')
if SHOW:
    plt.show()
else:
    plt.savefig(PATH + '/reward.png')
    plt.close()

###### Simulation ######
def sim(agent, env):
    state = env.reset()
    state = np.array(list(state.values()))
    H = list()
    NF = list()
    for step in range(STEPS):
        action = agent.get_action(state)
        #action = noise.get_action(action, step)
        new_state, reward, done = env.step(action) 
        new_state = np.array(list(new_state.values()))
        _, _, _, _, h, nf, _, _ = new_state
        H.append(h)
        NF.append(nf)
        state = new_state
    return H, NF

H, NF = sim(agent, env)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.plot(H, 'r')
ax1.set_xlabel('days')
ax1.set_ylabel('H')
ax2.plot(NF, 'b')
ax2.set_xlabel('days')
ax1.set_ylabel('NF')
if SHOW:
    plt.show()
else:
    plt.savefig(PATH + '/sim.png')
    plt.close()

data = pd.DataFrame(columns=('episode_reward', 'average_reward'))
data.episode_reward = rewards
data.average_reward = avg_rewards
data.to_csv(PATH +'/rewards.csv', index=False)

