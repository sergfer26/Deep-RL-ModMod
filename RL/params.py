PARAMS_ENV = {'R': 'min(0.01 * h, 10)' , \
              'P': '- (0.5/5) * (u3 + u4 + u7 + u9 + u10)' , \
              'STEP': 1/8, \
              'TIME_MAX': 90, \
              'FRECUENCY': 60, 
              'MONTH': '03'}

#Para entrenamiento MONTH puede ser '03','06','09' o 'RANDOM'
#Para benchmark y tournament es recomendable que sea 'RANDOM', pero no absolutamente necesario.

PARAMS_TRAIN = {'EPISODES': 500, \
                'STEPS': int(PARAMS_ENV['TIME_MAX']/PARAMS_ENV['STEP']), \
                'BATCH_SIZE': 128, \
                'SHOW': False, \
                'INDICE': 0} # Cuando es distinto de 0, fija un indice para simular
