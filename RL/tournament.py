from trainRL import sim,agent, env, noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import data_inputs
from get_indexes import Indexes

MONTHS = ['03','06','09','12']

'''
path_agent0 = 'results_ddpg/3_10_1425' #Agente que se entreno en marzo 
path_agent1 = 'results_ddpg/3_10_1429' #Agente que se entreno en junio 
path_agent2 = 'results_ddpg/3_12_1515' #Agente que se entreno en septiembre 
path_agent3 = 'results_ddpg/3_12_1518' #Agente que se entreno en diciembre 
'''

path_agent0 = 'results_ddpg/2_22_1449' #Agente que se entreno en marzo 
path_agent1 = 'results_ddpg/2_22_1449' #Agente que se entreno en junio 
path_agent2 = 'results_ddpg/2_22_1449' #Agente que se entreno en septiembre 
path_agent3 = 'results_ddpg/2_22_1449' #Agente que se entreno en diciembre 

AGENTS = [path_agent0,path_agent1,path_agent2,path_agent3]
noise.max_sigma = 0.0
noise.min_sigma = 0.0
number_of_simulations = 1
UMBRAL = 0.01

def get_score(agent, env, noise, month,path):
    agent.load(path) # Carga la red neuronal
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
    dic = {'mean_production':np.mean(production), 'var_production':np.var(production), 'mean_actions': promedios,\
        'var_actions': varianzas,'actions_above_umbral': acciones} 
    return dic

def get_score(agent, env, noise, month,path):
    X = np.random.uniform(0,1,2)
    dic = {'mean_production':X[0], 'var_production':X[1], 'mean_actions': np.random.uniform(0,1,10),\
        'var_actions': np.random.uniform(0,1,10),'actions_above_umbral': np.random.uniform(0,1,10) }
    return dic

def tournament(agent, env, noise):
    aux = {'mean_production':list(), 'var_production':list(), 'mean_actions': list(),\
        'var_actions': list(),'actions_above_umbral': list() }
    dictionary = dict.fromkeys(MONTHS, aux)
    for month in MONTHS:
        for path_agent in AGENTS:
            dic = get_score(agent, env, noise, month,path_agent)
            for k, v in dic.items():
                dictionary[month][k].append(v)
    breakpoint()
    fig, axes = plt.subplots(nrows=range(MONTHS))
    X = np.arange(10) #
    for ax, month in zip(axes, MONTHS):
        for x in X:
            ax.bar(X + 0.00, dictionary[month]['mean_actions'][:][0], color = 'b', width = 0.25)
    
if __name__ == '__main__':
    tournament(agent, env, noise)
    #get_score(agent, env, noise, '03',path_agent0)
