import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#index = 1
#data = pd.read_csv('index_'+str(index)+'.csv')

def filter_by_5_min(datos):
    matriz = np.ones((datos.shape[0]//5,4))
    for i in range(0, datos.shape[0]):
        if i % 5 == 0:
            matriz[i//5,:] = np.array(datos.loc[i])
    return pd.DataFrame(matriz,columns = datos.columns)


#data = filter_by_5_min(data)


def repeat_data(datos,repeticiones):
    datos1 = datos
    for i in range(repeticiones):
        datos = pd.concat([datos,datos1])
    return datos


#data = repeat_data(data,2)

    

