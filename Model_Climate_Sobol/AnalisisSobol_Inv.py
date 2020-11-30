from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from climate_env import ClimateU
import numpy as np
import pandas as pd
from time import time
#from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib 

def evaluate_model(A,dias):
    C1M, V1M, T1M, T2M =  ClimateU(A,k = 1440 * dias)
    return [C1M[-1], V1M[-1], T1M[-1], T2M[-1]]

def evaluate_model(A,dias):
    return np.random.uniform(0,1,4)

def datos(problem,n,dias,save):
    controles = saltelli.sample(problem, n,calc_second_order=True)
    data = pd.DataFrame(columns=("U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10","C1M", "V1M", "T1M", "T2M","Tiempo"))
    i = 0
    for control in controles:
        t1 = time()
        variables_de_estado =  list(evaluate_model(control,dias))
        t2 = time()
        control_y_variables = list(control)
        control_y_variables.extend(variables_de_estado)
        control_y_variables.append(round(t2-t1,4))
        data.loc[i] = control_y_variables
        i += 1
    if save:
        data.to_csv(str(len(data))+ '_observaciones_climaticas_'+str(dias)+'dias.csv' , index=False)
        print('Los datos se han guardado exitosamente')
    return data

def analisis(problem,datos,una_variable,save = False):
    nombre = 'imagen_Analisis_Sobol_'+una_variable+'.png'
    Y = np.array(datos[una_variable])
    Si = sobol.analyze(problem, Y,calc_second_order=True)
    fig, ax = plt.subplots(1)
    Si_filter = {k:Si[k] for k in ['ST','ST_conf','S1','S1_conf']}
    Si_df = pd.DataFrame(Si_filter, index=problem['names']) 
    indices = Si_df[['S1','ST']]
    err = Si_df[['S1_conf','ST_conf']]
    indices.plot.bar(yerr=err.values.T,ax=ax)
    fig.set_size_inches(8,4)
    plt.title(una_variable)
    if save:
        Si_df.to_csv('Analisis_Sobol_'+una_variable+'.csv')
        plt.savefig('imagen_'+'Analisis_Sobol_'+una_variable+'.png',dpi=200)
    else:
        plt.show()

def analizar_todo(problem,datos,save):
    for str_variable in ["C1M", "V1M", "T1M", "T2M"]:
        analisis(problem,datos,str_variable,save = save)
        print('Analisis de ' + str_variable + ' terminado' )
    

def main():
    problem = {"num_vars": 10,"names": ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10"],
               "bounds":  [[x,y] for x,y in zip(0*np.ones(10),np.ones(10))]}
    N = 50
    dias = 2
    informacion = datos(problem,N,dias,save = True)
    informacion = pd.read_csv('11000_observaciones_climaticas_2dias.csv')
    analizar_todo(problem,informacion,True)

if __name__ == "__main__":
    main()





