from numpy.lib.histograms import histogram
from sympy.utilities.iterables import bracelets
from trainRL import sim, noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import data_inputs, GreenhouseEnv 
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

number_of_simulations = 100
number_of_process     = 16
#path = sys.argv[1]
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
#agent.load('results_ddpg/' + path + '/nets')
agent_random = OtherAgent(env, 'random')
agent_on = OtherAgent(env, 'on')


def sim_(agent,env): return sim(agent, env)

def get_score(month,agent):                             
    p = multiprocessing.Pool(number_of_process) #Numero de procesadores
    indexes = Indexes(data_inputs[0:LIMIT],month)
    V = list()
    for _ in range(number_of_simulations):
        env = GreenhouseEnv() #Se crea el ambiente 
        env.indexes = indexes # Se crean indices
        V.append([agent,env])
    BIG_DATA = p.starmap(sim_, V)
    BIG_DATA = list(BIG_DATA)
    result = {'episode_rewards':list()}
    for i in range(1,12):result['u_'+str(i)] = list()
    for simulation in BIG_DATA:
        _, _, S_prod, A, _, _ = simulation
        df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
        dfa = pd.DataFrame(A, columns=['u_' + str(i) for i in range(1,12)])
        episode_reward = df_prod['$Cr_t$'].iloc[-1]
        result['episode_rewards'].append(episode_reward)
        for i in range(1,12): result['u_'+str(i)] += list(dfa['u_'+str(i)])
    return result

def save_score(PATH,result,name):
    name = PATH + '/simulations_' + name + '.json'
    if name != 'nn':
        result1 = {'episode_rewards':result['episode_rewards']}
    else:
        result1 = result
    with open(name, 'w') as fp:
        json.dump(result1, fp,  indent=4)

'''
def fig_production(string):
    PROMEDIOS = list()
    VARIANZAS = list()
    for name in NAMES:
        promedios = list()
        varianzas = list()
        for month in MONTHS:
            with open(PATH + '/'+month+'_'+name+'.json') as f:
                data = json.load(f)
            promedios.append(data['mean_' + string ])
            varianzas.append(data['var_' + string ])
        PROMEDIOS.append(promedios)
        VARIANZAS.append(varianzas)
    fig, axes = plt.subplots(nrows=2)
    X = np.arange(len(MONTHS))
    axes[0].set_ylabel('mean')
    axes[1].set_ylabel('var')
    axes[0].bar(X + 0.0, PROMEDIOS[0],  color = 'b', width = 0.15,label = NAMES[0] + ' = ' + str(float(np.round(PROMEDIOS[0],3))))
    axes[1].bar(X + 0.0, VARIANZAS[0],  color = 'b', width = 0.15, label =  NAMES[0])
    axes[0].bar(X + 0.15, PROMEDIOS[1],  color = 'g', width = 0.15, label =  NAMES[1] + ' = '+ str(float(np.round(PROMEDIOS[1],3))))
    axes[1].bar(X + 0.15, VARIANZAS[1],  color = 'g', width = 0.15, label =  NAMES[1])
    axes[0].bar(X + 0.30, PROMEDIOS[2],  color = 'r', width = 0.15, label =  NAMES[2] + ' = '+ str(float(np.round(PROMEDIOS[2],3))))
    axes[1].bar(X + 0.30, VARIANZAS[2],  color = 'r', width = 0.15, label =  NAMES[2])
    axes[0].bar(X + 0.45, PROMEDIOS[3],  color = 'c', width = 0.15, label =  NAMES[3] + ' = '+ str(float(np.round(PROMEDIOS[3],3))))
    axes[1].bar(X + 0.45, VARIANZAS[3],  color = 'c', width = 0.15, label =  NAMES[3])
    axes[1].set_xticks(X)
    axes[1].set_xticklabels(MONTHS)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=4, fancybox=True, shadow=True)
    axes[0].set_xticks(X)
    axes[0].set_xticklabels(MONTHS)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=4, fancybox=True, shadow=True)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.suptitle( string + ' (mean/var)', fontsize=15)
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/scores_'+ string +'.png')
        plt.close()


def fig_actions(key):
    fig, axes = plt.subplots()
    names = ['$u_' + str(x + 1) + '$' for x in range(9)]
    names.append('$u_{10}$')
    X = np.arange(len(names))
    for i,month in enumerate(MONTHS):
        LISTA = list()
        for name in NAMES:
            with open(PATH + '/'+month+'_' + name + '.json') as f:
                data = json.load(f)
                #dic[month].append(data[key])
                LISTA.append(data[key])
        axes.set_ylabel(datetime(2018, int(month), 1).strftime("%b"))
        axes.bar(X + 0.0, LISTA[0],  color = 'b', width = 0.15,label =  NAMES[0])
        axes.bar(X + 0.15, LISTA[1],  color = 'g', width = 0.15, label =  NAMES[1])
        axes.bar(X + 0.30, LISTA[2],  color = 'r', width = 0.15, label =  NAMES[2])
        axes.bar(X + 0.45, LISTA[3],  color = 'c', width = 0.15, label =  NAMES[3])
        axes.set_xticks(X)
        axes.set_xticklabels(names)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=4, fancybox=True, shadow=True)
    fig.suptitle(key.upper(), fontsize=15)
    fig.set_size_inches(18.5, 10.5, forward=True)
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/'+key+'.png')
        plt.close()


def histograms(key,same_x = False):
    fig, axes = plt.subplots(4,sharey=True)
    m1 = 0
    m2 = 0
    for i,name in enumerate(NAMES):
        with open(PATH + '/'+mes+'_' + name + '.json') as f:
            data = json.load(f)
            data = data['vector_' + key]
            m1 = max(m1,max(data))
            m2 = min(m2,min(data))
            axes[i].hist(data,bins = 30)
            axes[i].set_ylabel(name.upper())
            axes[i].axvline(np.mean(data), color='k', linestyle='dashed', linewidth=1)
    if same_x: 
        for i in range(4): axes[i].set_xlim(m2,m1)
    fig.suptitle(key.upper(), fontsize=15)
    fig.set_size_inches(18.5, 10.5, forward=True)
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/'+'histograms_' + key + '.png')
        plt.close()  


def main1():
    for name in NAMES[1:]:
        shutil.copy('results_ddpg/tournament/Month_' + mes + '/' + mes + '_' + name + '.json', PATH)
    get_score(mes,agent,'nn')

    histograms('reward')
    fig_actions('mean_actions')
    fig_actions('var_actions')
    histograms('mass',1)
    histograms('number_of_fruit',1)
    create_report(PATH,mes)
    for name in NAMES[1:]:
        os.remove(PATH + '/' + mes + '_' + name + '.json' )
    shutil.copy(PATH + '/Reporte_agentes.pdf', 'results_ddpg/' + path)
    shutil.copy(PATH + '/' + mes + '_nn.json', 'results_ddpg/' + path)

    shutil.copy(PATH + '/histograms_mass.png', 'results_ddpg/' + path)
    shutil.copy(PATH + '/histograms_number_of_fruit.png', 'results_ddpg/' + path)
    shutil.copy(PATH + '/histograms_reward.png' , 'results_ddpg/' + path)
    shutil.copy(PATH + '/mean_actions.png', 'results_ddpg/' + path)
    shutil.copy(PATH + '/var_actions.png', 'results_ddpg/' + path)

    os.remove(PATH + '/Reporte_agentes.pdf')
    breakpoint()
'''
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

if __name__ == '__main__':
    season1()