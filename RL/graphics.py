import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from params import PARAMS_TRAIN
import pytz
from datetime import datetime, timezone
from time import time
import pathlib
SHOW = PARAMS_TRAIN['SHOW']
def create_path():
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute
    PATH = 'results_ddpg/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
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
        fig.savefig(PATH + '/reward.png')
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
        plt.savefig(PATH + '/sim_climate.png')
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
        plt.savefig(PATH + '/sim_rh_par.png')
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
        plt.savefig(PATH + '/sim_prod.png')
        plt.close()

def figure_actions(A,final_indexes,dim,PATH):
    dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$', r'$u_{11}$'))
    title = 'Controles' # $U$
    dfa.index = final_indexes
    ax = dfa.plot(subplots=True, layout=(int(np.ceil(dim / 2)), 2), figsize=(10, 7), title=title) 
    for a in ax.tolist():a[0].set_ylim(0,1);a[1].set_ylim(0,1)
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/sim_actions.png')
        plt.close()

def figure_inputs(df_inputs,final_indexes,PATH):
    df_inputs.index = final_indexes
    ax = df_inputs.plot(subplots=True, figsize=(10, 7),title = 'Datos climaticos')
    ax[0].set_ylabel('$W*m^{2}$')
    ax[1].set_ylabel('C')
    ax[2].set_ylabel('$Km*h^{-1}$')
    ax[3].set_ylabel('$W*m^{2}$')
    ax[4].set_ylabel('%')
    plt.gcf().autofmt_xdate()
    if SHOW:
        plt.show()
        plt.close()
    else:
        plt.savefig(PATH + '/sim_climate_inputs.png')
        plt.close()
    
def save_Q(env,PATH):
    name = PATH + '/costos.json'
    with open(name, 'w') as fp:
        json.dump(env.Qvar_dic, fp,  indent=4)

def save_rewards(rewards, avg_rewards, penalties, abs_rewards,PATH):
    dic_rewards = {'rewards':rewards, 'avg_rewards': avg_rewards,'penalties': penalties,'abs_reward':abs_rewards}
    name = PATH + '/rewards.json'
    with open(name, 'w') as fp:
        json.dump(dic_rewards, fp,  indent=4)

def compute_indexes(inicio,step,time_max):
    data_inputs = pd.read_csv('Inputs_Bleiswijk.csv')
    for_indexes = int(step*24) 
    num_steps = int(1/step)*time_max
    new_indexes = [inicio+(for_indexes*j) for j in range(num_steps)]
    final_indexes = [data_inputs['Date'][index] for index in new_indexes]
    return final_indexes