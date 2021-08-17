minutos = 60
PARAMS_ENV = {'STEP': minutos/(24*60), \
              'TIME_MAX': 90, #Dias simulados\ 
              'FRECUENCY': 60, 
              'SEASON':1}

#Para entrenamiento SEASON puede ser 1,2 o 'RANDOM'
#Para benchmark y tournament es recomendable que sea 'RANDOM', pero no absolutamente necesario.

#El min de STEP  no es 1/24, pero el min de FRECUENCY SÍ es 60
PARAMS_TRAIN = {'EPISODES': 4000, \
                'STEPS': int(PARAMS_ENV['TIME_MAX']/PARAMS_ENV['STEP']), \
                'BATCH_SIZE': 128, \
                'SHOW': False, \
                'INDICE': 0, # Cuando es distinto de 0, fija un indice para simular
                'SAVE_FREQ': 1000
                } 

CONTROLS = {'u_1': 1, 
            'u_2': 1, 
            'u_3': 1, 
            'u_4': 1,   
            'u_5': 1, 
            'u_6': 1, 
            'u_7': 1, 
            'u_8': 1,  
            'u_9': 1, 
            'u_10': 1,
            'u_11': 1, 
            }
