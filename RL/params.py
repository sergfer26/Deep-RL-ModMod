import json
from climate_model.constants import OTHER_CONSTANTS
minutos = 60
PARAMS_ENV = {'STEP': minutos/(24*60), \
              'TIME_MAX': 5, #Dias simulados\ 
              'FRECUENCY': 60, 
              'SEASON':1, \
              'MINUTOS':minutos}

#Para entrenamiento SEASON puede ser 1,2 o 'RANDOM'
#Para benchmark y tournament es recomendable que sea 'RANDOM', pero no absolutamente necesario.

#El min de STEP  no es 1/24, pero el min de FRECUENCY SÍ es 60
PARAMS_TRAIN = {'EPISODES': 1000, \
                'STEPS': int(PARAMS_ENV['TIME_MAX']/PARAMS_ENV['STEP']), \
                'BATCH_SIZE': 128, \
                'SHOW': False, \
                'SERVER':False, \
                'INDICE': 0, # Cuando es distinto de 0, fija un indice para simular
                'SAVE_FREQ': 1000
                } 

PARAMS_SIM = {'anio':2017,\
            'mes':8,\
            'dia': 15,\
            'hora': 1
            }
            
CONTROLS = {'u_1': 1, 
            'u_2': 1, 
            'u_3': 1, 
            'u_4': 1,   
            'u_5': 1, #No hace nada por los parametros
            'u_6': 1, 
            'u_7': 1, 
            'u_8': 1,  
            'u_9': 1, 
            'u_10': 1,
            'u_11': 1, 
            }
f = open ('ddpg/parametros.json', "r")
parametros_ddpg = json.loads(f.read())

all_params = {
            'PARAMS_DDPG':parametros_ddpg['PARAMS_DDPG'], \
            'PARAMS_UTILS':parametros_ddpg['PARAMS_UTILS'], \
            'PARAMS_ENV': PARAMS_ENV, \
            'PARAMS_TRAIN':PARAMS_TRAIN,\
            'PARAMS_SIM':PARAMS_SIM, \
            'MODEL_NOISE':OTHER_CONSTANTS['model_noise'].val,\
            'CONTROLS': CONTROLS}
            
def save_params(dicc,path):
    with open(path + '/parametros.json', 'w') as outfile:
        json.dump(dicc, outfile,indent=4)


