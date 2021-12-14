from env import GreenhouseEnv, STEP, TIME_MAX
import numpy as np
from params import PARAMS_TRAIN,all_params, save_params
from tqdm import tqdm
STEPS = PARAMS_TRAIN['STEPS']
SERVER = PARAMS_TRAIN['SERVER']
if SERVER: SHOW = False #No puedes mostrar nada en el servidor

def sim(agent,ind = None):
    env = GreenhouseEnv() 
    if ind != None:
        env.i = ind
    start = env.i 
    state = np.array(list(env.state.values()))
    print('Se simula con indice = ', env.i)
    S_climate = np.zeros((STEPS, 4)) # vars del modelo climatico T1, T2, V1, C1
    S_data = np.zeros((STEPS, 2)) # datos recopilados RH PAR
    S_prod = np.zeros((STEPS, 6)) # datos de produccion h, nf, H, N, r_t, Cr_t
    A = np.zeros((STEPS, env.action_space.shape[0]))
    episode_reward = 0.0
    with tqdm(total=STEPS, position=0) as pbar:
        for step in range(STEPS):
            action = agent.get_action(state) 
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
            pbar.set_postfix(step='{}'.format(step),V1 = '{}'.format(V1))
            pbar.update(1)
    data_inputs = env.return_inputs_climate(start)
    return S_climate, S_data, S_prod, A, data_inputs,start