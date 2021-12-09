import pytz
import numpy as np
import pathlib
import pandas as pd
from datetime import datetime, timezone
from time import time
from simulation_date import get_index
from tqdm import tqdm
from trainRL import STEPS, action_dim, sim
from get_report_constants import Constants
from baseline_policy import agent_baseline
from matplotlib import pyplot as plt
from env import STEP, TIME_MAX, GreenhouseEnv,data_inputs
from params import minutos,PARAMS_TRAIN,PARAMS_SIM,save_params,all_params
from graphics import date,compute_indexes,figure_actions,figure_inputs,figure_prod,figure_rh_par,figure_state
from ddpg.ddpg import DDPGagent

SHOW = PARAMS_TRAIN['SHOW']
PATH = 'Simulaciones/'+ date() 
pathlib.Path(PATH+'/images').mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH+'/reports').mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH+'/data').mkdir(parents=True, exist_ok=True)
env = GreenhouseEnv()

agent = agent_baseline()

#NN control 
#agent = DDPGagent(env)
#agent.load('results_ddpg/8_17_1848/nets','_1000')


def sim_pid(agent, env, indice = 0):
    dt = 60/minutos
    state = env.reset() 
    start = env.i if indice == 0 else indice # primer indice de los datos
    env.i = start 
    S_climate = np.zeros((STEPS, 4)) # vars del modelo climatico T1, T2, V1, C1
    S_data = np.zeros((STEPS, 2)) # datos recopilados RH PAR
    S_prod = np.zeros((STEPS, 6)) # datos de produccion h, nf, H, N, r_t, Cr_t
    A = np.zeros((STEPS, action_dim))
    episode_reward = 0.0
    with tqdm(total=STEPS, position=0) as pbar:
        for step in range(STEPS):
            I5 = env.dirClimate.V('I5')
            #pbar.update(step)
            if step % dt == 0:
                indice1 = step/dt # una hora
            action = np.zeros(11) if step == 0 else agent.get_action(indice1, env)
            new_state, reward, done = env.step(action)
            episode_reward += reward
            C1, RH, T2, PAR, h, n = state
            T1 = env.dirClimate.V('T1'); V1 = env.dirClimate.V('V1')
            S_climate[step, :] = np.array([T1, T2, V1, C1]) 
            H = env.dirGreenhouse.V('H'); NF= env.dirGreenhouse.V('NF')
            S_data[step, :] = np.array([RH, PAR])
            S_prod[step, :] = np.array([h, n, H, NF, reward, episode_reward])
            pbar.set_postfix(step='{}'.format(step),I5 = '{}'.format(round(I5,2)))
            pbar.update(1)
            A[step, :] = action
            state = new_state
    data_inputs = env.return_inputs_climate(start)
    return S_climate, S_data, S_prod, A, data_inputs,start

def main():
    y = PARAMS_SIM['anio']
    m = PARAMS_SIM['mes']
    d = PARAMS_SIM['dia']
    h = PARAMS_SIM['hora']
    print('Buscando indice')
    save_params(all_params,PATH+'/reports')
    Constants(PATH)
    date = datetime(y,m,d,h)
    #ind = get_index(data_inputs,date)
    ind = 12311
    S_climate, S_data, S_prod, A, df_inputs,start = sim_pid(agent, env, indice = ind)
    start = df_inputs['Date'].iloc[0]
    final_indexes = compute_indexes(start,STEPS,env.frec)
    df_climate = pd.DataFrame(S_climate, columns=('$T_1$', '$T_2$', '$V_1$', '$C_1$'))
    df_climate.index = final_indexes
    df_climate.to_csv(PATH+'/data/' + 'climate_model.csv')
    figure_state(S_climate,final_indexes,PATH)
    figure_rh_par(S_data,final_indexes,PATH)
    figure_prod(S_prod,final_indexes,PATH)
    figure_actions(A,final_indexes,PATH)
    figure_inputs(df_inputs,PATH)
    print('Guardado en ',PATH)
    
if __name__ == '__main__':
    main()
