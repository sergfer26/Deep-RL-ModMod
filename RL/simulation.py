import pytz
import numpy as np
import pathlib
import pandas as pd
from datetime import datetime, timezone
from time import time
from progressbar import*
from trainRL import env, STEPS, action_dim
from baseline_policy import agent_baseline
from matplotlib import pyplot as plt
from env import  STEP, TIME_MAX
'''
tz = pytz.timezone('America/Mexico_City')
mexico_now = datetime.now()
month = mexico_now.month
day = mexico_now.day
hour = mexico_now.hour
minute = mexico_now.minute

'''
PATH = 'results_ddpg/BORRAME'#+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SHOW  = False
agent = agent_baseline(env)
data = pd.read_csv('Inputs_Bleiswijk.csv')

def sim(agent, env, indice = 0):
    pbar = ProgressBar(maxval=STEPS)
    pbar.start()
    state = env.reset() 
    start = env.i if indice == 0 else indice # primer indice de los datos
    env.i = start 
    S_climate = np.zeros((STEPS, 4)) # vars del modelo climatico T1, T2, V1, C1
    S_data = np.zeros((STEPS, 2)) # datos recopilados RH PAR
    S_prod = np.zeros((STEPS, 6)) # datos de produccion h, nf, H, N, r_t, Cr_t
    A = np.zeros((STEPS, action_dim))
    episode_reward = 0.0
    for step in range(STEPS):
        pbar.update(step)
        action = np.zeros(11) if step == 0 else agent.get_action(state,data['I5'][env.i])
        new_state, reward, done = env.step(action)
        episode_reward += reward
        C1, RH, T2, PAR, h, n = state
        T1 = env.dirClimate.V('T1'); V1 = env.dirClimate.V('V1')
        S_climate[step, :] = np.array([T1, T2, V1, C1]) 
        H = env.dirGreenhouse.V('H'); NF= env.dirGreenhouse.V('NF')
        S_data[step, :] = np.array([RH, PAR])
        S_prod[step, :] = np.array([h, n, H, NF, reward, episode_reward])
        A[step, :] = action
        state = new_state
    pbar.finish()
    data_inputs = env.return_inputs_climate(start)
    return S_climate, S_data, S_prod, A, data_inputs,start


S_climate, S_data, S_prod, A, df_inputs,start = sim(agent, env, indice = 0)

for_indexes = int(STEP*24) 
num_steps = int(1/STEP)*TIME_MAX
new_indexes = [start+(for_indexes*j) for j in range(num_steps)]
final_indexes = [data['Date'][index] for index in new_indexes]

df_climate = pd.DataFrame(S_climate, columns=('$T_1$', '$T_2$', '$V_1$', '$C_1$'))

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

dfa = pd.DataFrame(A, columns=('$u_1$', '$u_2$', '$u_3$', '$u_4$', '$u_5$', '$u_6$', '$u_7$', '$u_8$', '$u_9$', r'$u_{10}$', r'$u_{11}$'))
title = 'Controles' # $U$
dfa.index = final_indexes
ax = dfa.plot(subplots=True, layout=(int(np.ceil(action_dim / 2)), 2), figsize=(10, 7), title=title) 
for a in ax.tolist():a[0].set_ylim(0,1);a[1].set_ylim(0,1)
plt.gcf().autofmt_xdate()
if SHOW:
    plt.show()
else:
    plt.savefig(PATH + '/sim_actions.png')
    plt.close()

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
else:
    plt.savefig(PATH + '/sim_climate_inputs.png')
    plt.close()