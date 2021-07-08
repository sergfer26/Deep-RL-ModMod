from numpy.lib.histograms import histogram
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

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute
PATH = 'results_ddpg/tournament/' + str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SHOW = False


NAMES = ['nn','random','on','off']
number_of_simulations = 100
path = sys.argv[1]
mes = sys.argv[2]
MONTHS = [mes]

def sim_(agent,env):
    print(os.getpid())
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
#AGENTS = [agent, agent_random, agent_on, agent_off]



def get_score(month,agent,name):                             
    production = []
    mass = []
    reward = []
    promedios = np.zeros(10)
    varianzas = np.zeros(10)
    number_of_process = 16
    p = multiprocessing.Pool(number_of_process)
    indexes = Indexes(data_inputs[0:LIMIT],month)
    V = list()
    for _ in range(number_of_simulations):
        env = GreenhouseEnv() #Se crea el ambiente 
        env.indexes = indexes # Se crean indices
        V.append([agent,env])
    BIG_DATA = p.starmap(sim_, V)
    BIG_DATA = list(BIG_DATA)
    for s in BIG_DATA:
        _, _, S_prod, A, _, _ = s
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
    dic = {'mean_number_of_fruit':np.mean(production), 'var_number_of_fruit':np.var(production), 'mean_actions': list(promedios),'var_actions': list(varianzas),'mean_mass': np.mean(mass), 'var_mass': np.var(mass),'mean_reward':np.mean(reward),'var_reward':np.var(reward),'vector_mass': mass,'vector_number_of_fruit':production,'vector_reward':reward}
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
            axes[i].hist(data)
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


if __name__ == '__main__':
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
 










