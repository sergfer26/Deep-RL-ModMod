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
from multiprocessing import Pool
from functools import partial
import sys
import os

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute
PATH = 'results_ddpg/tournament/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SHOW = False

MONTHS = ['03']
NAMES = ['nn','random','on','off']
number_of_simulations = 10
path = sys.argv[1]

def sim_(v):
    agent, env = v
    return sim(agent, env)

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
agent.load('results_ddpg/' + path)
agent_random = OtherAgent(env, 'random')
agent_on = OtherAgent(env, 'on')
agent_off = OtherAgent(env, 'off')
AGENTS = [agent, agent_random, agent_on, agent_off]



def get_score(month,agent,name):                                    
    production = []
    mass = []
    reward = []
    promedios = np.zeros(10)
    varianzas = np.zeros(10)
    p = Pool()
    V = list()
    indexes = Indexes(data_inputs[0:LIMIT],month)
    for _ in range(number_of_simulations):
        env = GreenhouseEnv() #Se crea el ambiente 
        env.indexes = indexes
        V.append([agent,env])
    BIG_DATA = list(p.map(sim_, V))
    for s in BIG_DATA:
        _, _, S_prod, A, _ = s
        df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
        aux = len(df_prod) - 1
        number_of_fruit =  df_prod['$N$'][aux]
        dry_mass =  df_prod['$H$'][aux]
        accumulated_reward = df_prod['$Cr_t$'][aux]
        production.append(number_of_fruit)
        mass.append(dry_mass)
        reward.append(accumulated_reward)
        dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$'))
        vector_aux2 = [] # para promedio de acciones
        vector_aux3 = [] # para varianza de acciones
        for c in dfa.columns:
            vector_aux2.append(np.mean(dfa[c]))
            vector_aux3.append(np.var(dfa[c]))
        promedios += vector_aux2
        varianzas += vector_aux3
    promedios /= number_of_simulations
    varianzas /= number_of_simulations
    dic = {'mean_number_of_fruit':np.mean(production), 'var_number_of_fruit':np.var(production), 'mean_actions': list(promedios),'var_actions': list(varianzas),'mean_mass': np.mean(mass), 'var_mass': np.var(mass),'mean_reward':np.mean(reward),'var_reward':np.var(reward),'vector_mass': mass,'vector_number_of_fruit':production}
    name = PATH + '/'+month+'_'+name+'.json'
    with open(name, 'w') as fp:
        json.dump(dic, fp,  indent=4)


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
    axes[0].bar(X + 0.0, PROMEDIOS[0],  color = 'b', width = 0.15,label = NAMES[0])
    axes[1].bar(X + 0.0, VARIANZAS[0],  color = 'b', width = 0.15, label =  NAMES[0])
    axes[0].bar(X + 0.15, PROMEDIOS[1],  color = 'g', width = 0.15, label =  NAMES[1])
    axes[1].bar(X + 0.15, VARIANZAS[1],  color = 'g', width = 0.15, label =  NAMES[1])
    axes[0].bar(X + 0.30, PROMEDIOS[2],  color = 'r', width = 0.15, label =  NAMES[2])
    axes[1].bar(X + 0.30, VARIANZAS[2],  color = 'r', width = 0.15, label =  NAMES[2])
    axes[0].bar(X + 0.45, PROMEDIOS[3],  color = 'c', width = 0.15, label =  NAMES[3])
    axes[1].bar(X + 0.45, VARIANZAS[3],  color = 'c', width = 0.15, label =  NAMES[0])
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
    fig, axes = plt.subplots(nrows=len(MONTHS)+1)
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
        axes[i].set_ylabel(datetime(2018, int(month), 1).strftime("%b"))
        axes[i].bar(X + 0.0, LISTA[0],  color = 'b', width = 0.15,label =  NAMES[0])
        axes[i].bar(X + 0.15, LISTA[1],  color = 'g', width = 0.15, label =  NAMES[1])
        axes[i].bar(X + 0.30, LISTA[2],  color = 'r', width = 0.15, label =  NAMES[2])
        axes[i].bar(X + 0.45, LISTA[3],  color = 'c', width = 0.15, label =  NAMES[3])

        axes[i].set_xticks(X)
        axes[i].set_xticklabels(names)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=4, fancybox=True, shadow=True)
    fig.suptitle(key, fontsize=15)
    fig.set_size_inches(18.5, 10.5, forward=True)
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/'+key+'.png')
        plt.close()


if __name__ == '__main__':
    for month in MONTHS:
        for agent,name in zip(AGENTS,NAMES):
            print(month,name)
            get_score(month,agent,name)
    fig_production('number_of_fruit')
    fig_production('mass')
    fig_production('reward')
    fig_actions('mean_actions')
    fig_actions('var_actions')
    create_report(PATH)







