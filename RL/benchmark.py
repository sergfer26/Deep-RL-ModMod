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

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute
PATH = 'results_ddpg/tournament/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SHOW = False

MONTHS = ['03','06','09','12']
number_of_simulations = 50 
path = sys.argv[1]

def sim_(v):
    agent,env = v
    return sim(agent,env)

class OtherAgent(object):

    def __init__(self, env, type_):
        self.num_actions = env.action_space.shape[0]
        self.type = type_ # random, on, off

    def get_action(self):

        if self.type == 'random':
            return np.random.RandomState().uniform(0, 1, self.num_actions)
        elif self.type == 'on':
            return np.ones(self.num_actions)
        elif self.type == 'off':
            return np.zeros(self.num_actions)


env = GreenhouseEnv()
agent = DDPGagent(env)
agent.load('results_ddpg/'+ path)
agent_random = OtherAgent(env, 'random')
agent_on = OtherAgent(env, 'on')
agent_off = OtherAgent(env, 'off')
AGENTS = [agent, agent_random, agent_on, agent_off]

def get_score(month,agent):
    env = GreenhouseEnv()                                 #Se crea el ambiente 
    env.indexes = Indexes(data_inputs[0:env.limit],month) #Se crean nuevos indices
    production = []
    mass = []
    promedios = np.zeros(10)
    varianzas = np.zeros(10)
    p = Pool(number_of_simulations)
    V = [[agent,env] for _ in range(number_of_simulations)]
    BIG_DATA = list(p.map(sim_, V))
    for s in BIG_DATA:
        _, _, S_prod, A, _ = s
        df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
        aux = len(df_prod) - 1
        number_of_fruit =  df_prod['$N$'][aux]
        dry_mass =  df_prod['$H$'][aux]
        production.append(number_of_fruit)
        mass.append(dry_mass)
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
    dic = {'mean_number_of_fruit':np.mean(production), 'var_number_of_fruit':np.var(production), 'mean_actions': list(promedios),\
        'var_actions': list(varianzas),'mean_mass': np.mean(mass), 'var_mass': np.var(mass)}
    
    with open(name, 'w') as fp:
        json.dump(dic, fp,  indent=4)

def fig_production(string)
    PROMEDIOS = list()
    VARIANZAS = list()
    for path in AGENTS:
        promedios = list()
        varianzas = list()
        for month in MONTHS:
            with open(PATH + '/'+month+'_'+path+'.json') as f:
                data = json.load(f)
            promedios.append(data['mean_' + string ])
            varianzas.append(data['var_' + string ])
        PROMEDIOS.append(promedios)
        VARIANZAS.append(varianzas)
    fig, axes = plt.subplots(nrows=2)
    X = np.arange(len(MONTHS))
    axes[0].set_ylabel('mean')
    axes[1].set_ylabel('var')
    axes[0].bar(X + 0.0, PROMEDIOS[0],  color = 'b', width = 0.15,label = 'agent0')
    axes[1].bar(X + 0.0, VARIANZAS[0],  color = 'b', width = 0.15, label = 'agent0')
    axes[0].bar(X + 0.15, PROMEDIOS[1],  color = 'g', width = 0.15, label = 'agent1')
    axes[1].bar(X + 0.15, VARIANZAS[1],  color = 'g', width = 0.15, label = 'agent1')
    axes[0].bar(X + 0.30, PROMEDIOS[2],  color = 'r', width = 0.15, label = 'agent2')
    axes[1].bar(X + 0.30, VARIANZAS[2],  color = 'r', width = 0.15, label = 'agent2')
    axes[0].bar(X + 0.45, PROMEDIOS[3],  color = 'c', width = 0.15, label = 'agent3')
    axes[1].bar(X + 0.45, VARIANZAS[3],  color = 'c', width = 0.15, label = 'agent3')
    axes[1].set_xticks(X)
    axes[1].set_xticklabels(MONTHS)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=4, fancybox=True, shadow=True)
    axes[0].set_xticks(X)
    axes[0].set_xticklabels(MONTHS)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=4, fancybox=True, shadow=True)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.suptitle('Number of fruits (mean/var)', fontsize=15)
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/scores_'+ string +''.png')
        plt.close()


def fig_actions(key):
    fig, axes = plt.subplots(nrows=len(MONTHS))
    names = ['$u_' + str(x + 1) + '$' for x in range(9)]
    names.append('$u_{10}$')
    X = np.arange(len(names))
    for i,month in enumerate(MONTHS):
        LISTA = list()
        for path in AGENTS:
            with open(PATH + '/'+month+'_'+path+'.json') as f:
                data = json.load(f)
                #dic[month].append(data[key])
                LISTA.append(data[key])
    
        axes[i].set_ylabel(datetime(2018, int(month), 1).strftime("%b"))
        axes[i].bar(X + 0.0, LISTA[0],  color = 'b', width = 0.15,label = 'agent0')
        axes[i].bar(X + 0.15, LISTA[1],  color = 'g', width = 0.15, label = 'agent1')
        axes[i].bar(X + 0.30, LISTA[2],  color = 'r', width = 0.15, label = 'agent2')
        axes[i].bar(X + 0.45, LISTA[3],  color = 'c', width = 0.15, label = 'agent3')

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

def create_json():
    for month in MONTHS:
        for agent in AGENTS:
            get_score(month,agent)


if __name__ == '__main__':
    create_json()
    fig_production('number_of_fruits')
    fig_actions('mean_actions')
    fig_actions('var_actions')
    fig_actions('actions_above_umbral')






