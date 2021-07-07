import numpy as np

######################### Funciones auxiliares de solver climate ########################
from solver_climate import a1,g1,h4,o2,h6,r6


#########################################################################################
q_gas     = 2.45  # Precio de gas natural
n_gas     = 35.26 # Efiencia energetica del gas natural
lambda4   = 5e5   # Capacidad calorifica del calentador de aire directo
alpha6    = 1e4   # Area de la supercie del piso del invernadero

q_co2_ext = 3.5 #El costo del gas de la fuente externa Mxm

def H_Boil_Pipe(r6,h4):
    return max(r6 + h4,0)

def G(H):
    ''' Ganancia en pesos por el peso H '''
    return 0.015341*H

def Q_gas(Clima):
    ''' Costo del combustible (Gas) '''
    h_6 = h6(U4=Clima.V('U4'), lamb4=Clima.V('lamb4'), alpha6=Clima.V('alpha6')) #H blow air 
    a_1 = a1(I1=Clima.V('I1'), beta3=Clima.V('beta3')) #auxiliar para g1
    g_1 = g1(a1=a_1)                                   #auxiliar para r6
    r_6 = r6(T1=Clima.Vk('T1'), I3=Clima.V('I3'), alpha3=Clima.V('alpha3'), epsil1=Clima.V('epsil1'), epsil2=Clima.V('epsil2'), lamb=Clima.V('lamb'), g1=g_1)
    h_4 = h4(T2=Clima.Vk('T2'), I3=Clima.V('I3'),gamma1=Clima.V('gamma1'), phi1=Clima.V('phi1'))
    H_boil_pipe = H_Boil_Pipe(r_6,h_4)
    Qgas = (q_gas/n_gas)*(H_boil_pipe + h_6)
    return Qgas

def Q_co2(Clima):
    '''Costo del CO_2'''
    o_2 = o2(U10=Clima.V('U10'), psi2=Clima.V('psi2'), alpha6=Clima.V('alpha6')) #MC_ext_air
    Qco2 = (10**6)*q_co2_ext*o_2
    return Qco2






