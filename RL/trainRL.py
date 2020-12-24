import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import GreenhouseEnv
from ddpg.ddpg import DDPGagent
from ddpg.utils import *
from tqdm import tqdm

EPISODES = 10
STEPS = 15
BATCH_SIZE = 128

#env = gym.make("Pendulum-v0")

def train_agent(agent, env, noise, steps=STEPS, episodes=EPISODES, batch_size=BATCH_SIZE):
    rewards = []
    avg_rewards = [] 
    for episode in range(EPISODES):
        with tqdm(total=STEPS, position=0) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
            state = env.reset()
            import pdb; pdb.set_trace()
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
                pbar.set_postfix(reward='{:.2f}'.format(episode_reward), NF='{:2f}'.format(NF), H='{:2f}'.format(H))
                pbar.update(1)      
        
                state = new_state
                episode_reward += reward

                if done:
                    #sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                    break
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    return rewards, avg_rewards

env = GreenhouseEnv()
agent = DDPGagent(env)
noise = OUNoise(env.action_space)


rewards_1, avg_rewards_1 = train_agent(agent, env, noise)
plt.plot(rewards_1, "--b", label='reward (DDPG)', alpha=0.1)
plt.plot(avg_rewards_1, "-b", label='avg reward (DDPG)')
plt.legend(loc="lower right")

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()