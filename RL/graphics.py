import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from params import PARAMS_TRAIN
import pytz
from datetime import datetime, timezone
from time import time
import pathlib
from env import ON_ACTIONS
SHOW = PARAMS_TRAIN['SHOW']

def date():
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    year = mexico_now.year
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute
    return str(year) +'_'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)

def create_path():
    PATH = 'results_ddpg/'+ date()
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    folders = ['/images','/output','/reports','/nets']
    for folder_name in folders:
        pathlib.Path(PATH + folder_name).mkdir(parents=True, exist_ok=True)
    return PATH

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def figure_reward(rewards, avg_rewards, penalties, abs_rewards,PATH):
    '''Grafica los rewards'''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    fig.suptitle('Rewards', fontsize=14)

    ax1.plot(rewards, "-b", label='reward (DDPG)',alpha = 0.3)
    ax1.plot(avg_rewards, "--b", label='avg reward (DDPG)', alpha=0.2)
    pts = 50
    ax1.plot(smooth(rewards,pts), color= 'indigo', label='Smooth reward DDPG', alpha=0.6)
    ax1.set_xlabel('episode')
    ax1.legend(loc='best')

    ax2.plot(abs_rewards, label='absolute reward', alpha=0.5)
    ax2.plot(penalties, label='penalty', alpha=0.5)
    ax2.set_xlabel('episode')
    ax2.legend(loc='best')
    if SHOW:
        plt.show()
        plt.close()
    else:
        fig.savefig(PATH + '/images/reward.png')
        plt.close()

def figure_state(S_climate,indexes,PATH):
    df_climate = pd.DataFrame(S_climate, columns=('$T_1$', '$T_2$', '$V_1$', '$C_1$'))
    df_climate.index = indexes
    ax = df_climate.plot(subplots=True, layout=(2, 2), figsize=(10, 7),title = 'Variables de estado') 
    ax[0,0].set_ylabel('$ ^{\circ} C$')
    ax[0,1].set_ylabel('$ ^{\circ} C$')
    ax[1,0].set_ylabel('Pa')
    ax[1,1].set_ylabel('$mg * m^{-3}$')
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/images/sim_climate.png')
        plt.close()
    
def figure_rh_par(S_data,final_indexes,PATH):
    df_data = pd.DataFrame(S_data, columns=('RH','PAR'))
    df_data.index = final_indexes
    ax = df_data.plot(subplots=True, layout=(1, 2), figsize=(10, 7),title = 'Promedios diarios') 
    ax[0,0].set_ylabel('%')
    ax[0,1].set_ylabel('$W*m^{2}$')
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/images/sim_rh_par.png')
        plt.close()

def figure_prod(S_prod,final_indexes,PATH):
    df_prod = pd.DataFrame(S_prod, columns=('$h$', '$nf$', '$H$', '$N$', '$r_t$', '$Cr_t$'))
    df_prod.index = final_indexes
    title= 'Produccion y recompensas'
    ax = df_prod.plot(subplots=True, layout=(3, 2), figsize=(10, 7), title=title) 
    ax[0,0].set_ylabel('g')
    ax[1,0].set_ylabel('g')
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/images/sim_prod.png')
        plt.close()

def figure_actions(A,final_indexes,PATH):
    columns = list()
    for i in ON_ACTIONS:
        columns.append('$U_{' + '{}'.format(i+1) + '}$' )
    dfa = pd.DataFrame(A, columns=columns)
    title = 'Controles' # $U$
    dfa.index = final_indexes
    ax = dfa.plot(subplots=True, layout=(int(np.ceil(len(ON_ACTIONS) / 2)), 2), figsize=(10, 7), title=title) 
    for a in ax.tolist():a[0].set_ylim(-0.1,1.1);a[1].set_ylim(-0.1,1.1)
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/images/sim_actions.png')
        plt.close()

def figure_inputs(df_inputs,PATH):
    df_inputs.index = df_inputs['Date']
    df_inputs.drop('Date', inplace=True, axis=1)
    ax = df_inputs.plot(subplots=True,layout=(3, 2), figsize=(10, 7),title = 'Datos climaticos')

    ax[0,0].set_ylabel('$W*m^{2}$')
    ax[0,1].set_ylabel('C')
    ax[1,0].set_ylabel('$Km*h^{-1}$')
    ax[1,1].set_ylabel('$W*m^{2}$')
    ax[2,0].set_ylabel('%')
    ax[2,1].set_ylabel('Pa')
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/images/sim_climate_inputs.png')
        plt.close()
    
def save_Q(dic,PATH):
    name = PATH + '/output/costos.json'
    with open(name, 'w') as fp:
        json.dump(dic, fp,  indent=4)

def save_rewards(rewards, avg_rewards, penalties, abs_rewards,PATH):
    dic_rewards = {'rewards':rewards, 'avg_rewards': avg_rewards,'penalties': penalties,'abs_reward':abs_rewards}
    name = PATH + '/output/rewards.json'
    with open(name, 'w') as fp:
        json.dump(dic_rewards, fp,  indent=4)

def compute_indexes(inicio,periods,frec):
    freq = str(frec)+'min'
    indexes = pd.date_range(inicio, periods=periods, freq=freq)
    return indexes

def figure_cost_gain(dic,final_indexes,PATH):
    columns = list(dic.keys())
    data = list()
    for name in columns:data.append(dic[name])
    data = np.array(data)
    data = data.T
    data_cost = pd.DataFrame(data,columns = columns)
    data_cost.index = final_indexes
    ax = data_cost.plot(subplots=True, layout=(2, 2), figsize=(10, 7), title='Ganancia y costos en $(mxn)m^{-2}min^{-1}$') 
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/images/sim_cost.png')
        plt.close()

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('')

def aux_llave(path,llave):
    f = open(path + '/output/simulations_nn.json','r') 
    data = json.load(f)
    x1 = data[llave]
    f = open(path + '/output/parametros.json','r')
    data = json.load(f)
    x2 = str(data['PARAMS_DDPG']['hidden_sizes'])
    return x1,x2

def aux_llave2(path,llave):
    f = open(path + '/output/rewards.json','r') 
    data = json.load(f)
    x1 = data[llave]
    f = open(path + '/output/parametros.json','r')
    data = json.load(f)
    x2 = str(data['PARAMS_DDPG']['hidden_sizes'])
    return x1,x2
def graficas_rendimiento():
    new_data = list()
    labels  = list()
    paths = ['2021_12_30_200','2021_12_30_205','2021_12_30_209','2021_12_30_2011',]
    path = 'results_ddpg/'
    for fecha in paths:
        x1, x2 = aux_llave(path + fecha,'episode_rewards')
        new_data.append(x1)
        labels.append(x2)
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data,showmeans=True)
    axis.set_title('Rentabilidad $mxn/m^2$')
    set_axis_style(axis, labels)
    #plt.show()
    plt.savefig('ImagesTesis/comparacion_entre_topologias')
    plt.close()


def graficas_rendimiento2():
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    new_data = list()
    labels  = list()
    paths = ['2021_12_30_200','2021_12_30_205','2021_12_30_209','2021_12_30_2011',]
    path = 'results_ddpg/'
    pts = 50
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    for fecha in paths:
        x1, x2 = aux_llave2(path + fecha,'rewards')
        axis.plot(smooth(x1,pts), alpha=0.7,label = x2)
    axis.plot(np.zeros_like(x1),color = 'k',linestyle='--')
    axis.set_xlabel('episodios')
    axis.set_ylabel(r'\textit{reward}')
    plt.legend()
    #plt.show()
    plt.savefig('ImagesTesis/rewards')
    plt.close()

def graficas_mass():
    new_data = list()
    labels  = list()
    paths = ['2021_12_30_200','2021_12_30_205','2021_12_30_209','2021_12_30_2011',]
    path = 'results_ddpg/'
    for fecha in paths:
        x1, x2 = aux_llave(path + fecha,'mass')
        new_data.append(x1)
        labels.append(x2)
    _, axis= plt.subplots(sharex=True, figsize=(10,5))
    axis.violinplot(new_data,showmeans=True)
    axis.set_title('Masa $g$')
    set_axis_style(axis, labels)
    #plt.show()
    plt.savefig('ImagesTesis/comparacion_entre_topologias_mass')
    plt.close()
if __name__ == '__main__':
    graficas_rendimiento()
    graficas_rendimiento2()
    graficas_mass()