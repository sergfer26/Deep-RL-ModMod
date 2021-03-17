from trainRL import sim,agent, env, noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import data_inputs
import glob 
import pytz
import json
import pathlib
from datetime import datetime, timezone
from get_indexes import Indexes
from get_report_agents import create_report

MONTHS = ['03','06','09','12']


path_agent0 = '3_10_1425' #Agente que se entreno en marzo 
path_agent1 = '3_10_1429' #Agente que se entreno en junio 
path_agent2 = '3_12_1515' #Agente que se entreno en septiembre 
path_agent3 = '3_12_1518' #Agente que se entreno en diciembre 

AGENTS = [path_agent0,path_agent1,path_agent2,path_agent3]
noise.max_sigma = 0.0
noise.min_sigma = 0.0
number_of_simulations = 1
UMBRAL = 0.01

tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now(tz)
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute
PATH = 'results_ddpg/tournament/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SHOW = False

def get_score(agent, env, noise, month,path):
    agent.load('results_ddpg/'+ path) # Carga la red neuronal
    env.indexes = Indexes(data_inputs[0:env.limit],month)
    production = []
    acciones = np.zeros(10)
    promedios = np.zeros(10)
    varianzas = np.zeros(10)
    for s in range(number_of_simulations):
        _, _, S_prod, A, _ = sim(agent, env, noise) 
        df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
        aux = len(df_prod) - 1
        number_of_fruit =  df_prod['$N$'][aux]
        production.append(number_of_fruit)

        dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$'))
        vector_aux1 = [] # para acciones arriba del umbral
        vector_aux2 = [] # para promedio de acciones
        vector_aux3 = [] # para varianza de acciones
        for c in dfa.columns:
            vector_aux1.append(sum(dfa[c] > UMBRAL))
            vector_aux2.append(np.mean(dfa[c]))
            vector_aux3.append(np.var(dfa[c]))

        acciones += vector_aux1
        promedios += vector_aux2
        varianzas += vector_aux3
    acciones /= number_of_simulations
    promedios /= number_of_simulations
    varianzas /= number_of_simulations
    dic = {'mean_production':np.mean(production), 'var_production':np.var(production), 'mean_actions': list(promedios),\
        'var_actions': list(varianzas),'actions_above_umbral': list(acciones)}
    
    with open(name, 'w') as fp:
        json.dump(dic, fp,  indent=4)

def get_score(agent, env, noise, month,path):
    X = np.random.uniform(0,1,2)
    dic = {'mean_production':X[0], 'var_production':X[1], 'mean_actions': list(np.random.uniform(0,1,10)),\
        'var_actions': list(np.random.uniform(0,1,10)),'actions_above_umbral': list(np.random.uniform(0,1,10)) }
    name = PATH + '/'+month+'_'+path+'.json'
    with open(name, 'w') as fp:
        json.dump(dic, fp,  indent=4)

def fig_production():
    PROMEDIOS = list()
    VARIANZAS = list()
    for path in AGENTS:
        promedios = list()
        varianzas = list()
        for month in MONTHS:
            with open(PATH + '/'+month+'_'+path+'.json') as f:
                data = json.load(f)
            promedios.append(data['mean_production'])
            varianzas.append(data['var_production'])
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
        plt.savefig(PATH + '/scores_productions.png')
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
        for path in AGENTS:
            get_score(agent, env, noise, month,path)
        

if __name__ == '__main__':
    create_json()
    fig_production()
    fig_actions('mean_actions')
    fig_actions('var_actions')
    fig_actions('actions_above_umbral')
    create_report(PATH)
