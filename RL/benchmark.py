from numpy.lib.histograms import histogram
from sympy.utilities.iterables import bracelets
from trainRL import sim, noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import data_inputs, GreenhouseEnv,ON_ACTIONS
from ddpg.ddpg import DDPGagent 
import glob 
import pytz
import json
import pathlib
from datetime import datetime, timezone
from get_indexes import Indexes
from get_report_agents import create_report
import multiprocessing
from functools import partial
import sys
import os
from time import time
import shutil
import os
import glob
from params import PARAMS_ENV
from baseline_policy import agent_baseline

number_of_simulations = 1
number_of_process     = 16
path = 'results_ddpg/' + sys.argv[1]

SEASON  =  PARAMS_ENV['SEASON']


class OtherAgent(object):
    def __init__(self, env, type_):
        self.num_actions = env.action_space.shape[0]
        self.type = type_ # random, on, off
    def get_action(self,state):
        if self.type == 'random':
            return np.random.RandomState().uniform(0, 1, self.num_actions)
        elif self.type == 'on':
            return np.ones(self.num_actions)
        elif self.type == 'off':
            return np.zeros(self.num_actions)


env = GreenhouseEnv()
LIMIT = env.limit
agent = DDPGagent(env)

agent_random = OtherAgent(env, 'random')
agent_on = OtherAgent(env, 'on')



def get_score(month,agent,sim):                             
    p = multiprocessing.Pool(number_of_process) #Numero de procesadores
    indexes = Indexes(data_inputs[0:LIMIT],month)
    V = list()
    for _ in range(number_of_simulations):
        env = GreenhouseEnv() #Se crea el ambiente 
        env.indexes = indexes # Se crean indices
        V.append([agent,env])
    BIG_DATA = p.starmap(sim, V)
    BIG_DATA = list(BIG_DATA)
    result = {'episode_rewards':list(),'mass':list()}
    for i in ON_ACTIONS:
        result['$U_{' + '{}'.format(i+1) + '}$' ] = list()
    #for i in range(1,12):result['u_'+str(i)] = list()
    for simulation in BIG_DATA:
        _, _, S_prod, A, _, _ = simulation
        df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
        dfa = pd.DataFrame(A, columns=['$U_{' + '{}'.format(i+1) + '}$' for i in ON_ACTIONS])
        episode_reward = df_prod['$Cr_t$'].iloc[-1]
        mass_reward    = df_prod['$H$'].iloc[-1] 
        result['episode_rewards'].append(episode_reward)
        result['mass'].append(mass_reward)
        for i in ON_ACTIONS:
            result['$U_{' + '{}'.format(i+1) + '}$' ] += list(dfa['$U_{' + '{}'.format(i+1) + '}$'])
        #for i in range(1,12): result['u_'+str(i)] += list(dfa['u_'+str(i)])
    return result

def save_score(PATH,result,name):
    name = PATH + '/output/simulations_' + name + '.json'
    with open(name, 'w') as fp:
        json.dump(result, fp,  indent=4)

def season1():
    '''Solo debe ejecutar una vez'''
    pathlib.Path('results_ddpg/tournament/Season1').mkdir(parents=True, exist_ok=True)
    score = get_score(1,agent_on)
    save_score('results_ddpg/tournament/Season1',score,'on')
    score = get_score(1,agent_random)
    save_score('results_ddpg/tournament/Season1',score,'random')

def season2():
    '''Solo debe ejecutar una vez'''
    pathlib.Path('results_ddpg/tournament/Season2').mkdir(parents=True, exist_ok=True)
    score = get_score(2,agent_on)
    save_score('results_ddpg/tournament/Season2',score,'on')
    score = get_score(2,agent_random)
    save_score('results_ddpg/tournament/Season2',score,'random')

def season1_nn(name = ''):
    agent.load(path + '/nets',name)
    score = get_score(1,agent,sim)
    save_score(path,score,'nn' + name)

def expert_control():
    '''
    Solo deberia ejecutarse una vez
    '''
    agent = agent_baseline()
    from simulation import sim
    score = get_score(1,agent,sim)
    save_score(path,score,'expert_1h')



def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('')

def violin_actions(name):
    f = open(path + '/output/simulations_'+name+'.json') 
    data = json.load(f)
    new_data = list()
    for i in ON_ACTIONS:
        new_data.append(data['$U_{' + '{}'.format(i+1) + '}$'])
        #new_data.append(data['u_' + str(i)])
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data)
    axis.set_title('Controles')
    labels = list()
    for i in ON_ACTIONS:
        labels.append('$U_{' + '{}'.format(i+1) + '}$' )
    set_axis_style(axis, labels)
    plt.savefig(path + '/images/violin_actions.png')
    plt.close()

def violin_reward(name):
    f = open(path + '/output/simulations_'+name+'.json') 
    data = json.load(f)
    new_data = list()
    new_data.append(data['episode_rewards'])

    f = open('results_ddpg/expert_control/output/simulations_expert_1h.json','r') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])
    '''
    f = open('results_ddpg/tournament/Season1/simulations_on.json','r') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])

    f = open('results_ddpg/tournament/Season1/simulations_random.json','r') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])
    '''
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data,showmeans=True)
    axis.set_title('Rentabilidad $mxn/m^2$')
    labels = [name]
    labels += ['expert']
    set_axis_style(axis, labels)
    plt.savefig(path + '/images/violin_rewards.png')
    plt.close()


def violin_reward_nets(names):
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    new_data = list()
    names= [str(name) for name in names]
    for name in names:
        f = open(path + '/output/simulations_nn_'+name+'.json') 
        data = json.load(f)
        new_data.append(data['episode_rewards'])    
    axis.violinplot(new_data, showmeans=True)
    axis.set_title('Rentabilidad $mxn/m^2$')
    labels = names
    set_axis_style(axis, labels)
    plt.savefig(path + '/images/violin_rewards_5000e.png')
    #plt.show()
    plt.close()


def ESPECIAL():
    new_data = list()
    f = open('results_ddpg/Redes_Sergio/output/simulations_nn_r_0.json','r') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])

    f = open('results_ddpg/8_17_1848/output/simulations_nn_1h.json') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])

    f = open('results_ddpg/8_17_1851/output/simulations_nn.json') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])
    
    f = open('results_ddpg/8_17_1853/output/simulations_nn.json') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])

    f = open('results_ddpg/expert_control/output/simulations_expert_1h.json','r') 
    data = json.load(f)
    new_data.append(data['episode_rewards'])
    
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data,showmeans=True)
    axis.set_title('Rentabilidad $mxn/m^2$')
    labels = ['64x64r','64x64x64','128x128','256x256','expert']
    set_axis_style(axis, labels)
    plt.show()
    plt.close()


def ESPECIAL1():
    new_data = list()

    f = open('results_ddpg/8_17_1848/output/simulations_nn.json') 
    data = json.load(f)
    new_data.append(data['mass'])

    f = open('results_ddpg/expert_control/output/simulations_expert.json','r') 
    data = json.load(f)
    new_data.append(data['mass'])
    
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data,showmeans=True)
    #axis.set_title('Rentabilidad $mxn/m^2$')
    axis.set_title('Masa $g$')
    labels = ['64x64x64','expert']
    set_axis_style(axis, labels)
    plt.show()
    plt.close()

if __name__ == '__main__':
    pass
    #expert_control()
    season1_nn('')
    violin_reward('nn') ##puede ser nn ó expert
    violin_actions('nn')
    #violin_reward_nets([1,1000,2000,3000,4000])
    #ESPECIAL1()
