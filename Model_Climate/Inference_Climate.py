#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:21:32 2020

@author: jdmolinam
"""

##################################################################
##################################################################
################# Inferencia - Modelo Climático ##################
##################################################################
##################################################################


###########################
####### Módulos  ##########
###########################
from Solver import Sol_Climate, OutData
from math import sqrt, exp, log, pi
from scipy.stats import gamma
import pytwalk
from numpy import array, mean, median, argmin, arange, savetxt, zeros, loadtxt, quantile
from matplotlib.pyplot import hist, figure, xlabel, ylabel, axvline, plot, title


##################################################################
################ Ejecución del MCMC ##############################
##################################################################

Data = OutData(nmrec=2*1440)
C1 = Data[0]
V1 = Data[1]
T2 = Data[2]


##################################################################
################ Ejecución del MCMC ##############################
##################################################################

### Soporte del parámetro ###
def ClimateSupp(theta):
    """
    Se evalúa que un candidato para la a posteriori de theta
    efectivamente esté en el soporte del parámetro.
    theta = [alpha1, phi2, psi2].
    """
    rt = True
    rt &= (theta[0] > 0)
    rt &= (theta[1] > 0)
    rt &= (theta[2] > 0)
    return rt

### Valores iniciales ###
def ClimateInit():
    """
    Genera valores iniales para theta 
    a partir de su distribución a priori.
    theta = [alpha1, phi2, psi2, sig_T2, sig_V1, sig_C1].
    """
    alpha1 = gamma.rvs(a=0.01, scale=1/3.33e-6)
    phi2 = gamma.rvs(a=0.01, scale=1/5e-4)
    psi2 = gamma.rvs(a=0.01, scale=1/4.34783e-8)
    return array( [alpha1, phi2, psi2] )

### Función energía ###
def ClimateU(theta):
    """
    Esta es la función energía:
    U = - log [ f (theta | Datos) ].
    theta = [alpha1, phi2, psi2].    
    """
    ## Se soluciona el sistema para nmrec minutos
    k = 2*1440
    Res = Sol_Climate(theta=theta, nmrec=k)
    C1M = Res[0]
    V1M = Res[1]
    T2M = Res[3]
    ## Cálculo de la energía
    U = 0.99*( log(theta[0]) + log(theta[1]) + log(theta[2]) ) \
    + (1 / (2*0.08**2))*sum( (T2M - T2)**2 ) + (1 / (2*0.08**2))*sum( (V1M - V1)**2 ) + (1 / (2*5**2))*sum( (C1M - C1)**2 ) \
    + 3.3e-6*theta[0] + 5e-4*theta[1] + 4.34783e-8*theta[2]
    return U

## Ejecución del MCMC ###
#Init0 = array([1000, 5, 1.3e5])    
#Init1 = array([5000, 40, 3.3e5]) # Init0 y Init1 son valores iniciales cercanos a los nominales
Init0 = array([2000, 15, 1.9e5])    
Init1 = array([4000, 25, 2.7e5]) # Init0 y Init1 son valores iniciales cercanos a los nominales
'''
Climate = pytwalk.pytwalk( n=3, U=ClimateU, Supp=ClimateSupp)
Climate.Run( T=3, x0=Init0, xp0=Init1) # ¿T? ¿x0? ¿xp0?
import pdb; pdb.set_trace()

# Análsis resultados
#Climate.Ana() 
## Extracción de resultados
# param = Climate.Output() # Las primeras columnas son los paramétros y la última los valores de la energía ## no estaba comenentado
Climate.Save('param_1k') # estaba comentado
'''
param = loadtxt('param_1k') # estaba comentado
alpha1 = param[: , 0]
phi2 = param[: , 1]
psi2 = param[: , 2]
U = param[: , 3]

## Se guardan los resultados  -- Los datos pueden cargarse con la funcion loadtxt de numpy
savetxt('alpha1_1k', alpha1)
savetxt('phi2_1k', phi2)
savetxt('psi2_1k', psi2)
savetxt('U_1k', U)

## Estimadores a posteriori
# media a posteriori
alpha1_mn = mean(alpha1) 
phi2_mn = mean(phi2) 
psi2_mn = mean(psi2) 
theta_mn = array([alpha1_mn, phi2_mn, psi2_mn])
# mediana a posteriori
alpha1_md = median(alpha1) 
phi2_md = median(phi2) 
psi2_md = median(psi2) 
theta_md = array([alpha1_md, phi2_md, psi2_md])
# MAP
idMax = argmin(U) # El mínimo de la energía es el máximo de la log posterior, pues U = - log (post)
alpha1_map = alpha1[idMax] 
phi2_map = phi2[idMax] 
psi2_map = psi2[idMax] 
theta_map = array([alpha1_map, phi2_map, psi2_map])

### Histogramas
hist(alpha1, density=True)
axvline(x=3000, color='red')
xlabel(r'$\alpha_1$')
ylabel("Density")
figure()
hist(phi2, density=True)
axvline(x=20, color='red')
xlabel(r'$\phi_2$')
ylabel("Density")
figure()
hist(psi2, density=True)
axvline(x=2.3e5, color='red')
xlabel(r'$\psi_2$')
ylabel("Density")

## Soluciones con los estimadores posteriores
# Solución con la media
Res_mn = Sol_Climate(theta=theta_mn, nmrec=5*1440, Inf=False) # No se incluye el tiempo de inferencia y se usa la base de predicción
C1_mn = Res_mn[0]
V1_mn = Res_mn[1]
T2_mn = Res_mn[3]
V1_mn /= 7 # Para re-escalar los resultados a una escala realista
# Solución con la mediana
Res_md = Sol_Climate(theta=theta_md, nmrec=5*1440, Inf=False)
C1_md = Res_md[0]
V1_md = Res_md[1]
T2_md = Res_md[3]
V1_md /= 7 # Para re-escalar los resultados a una escala realista
# Solución con el map
Res_map = Sol_Climate(theta=theta_map, nmrec=5*1440, Inf=False)
C1_map = Res_map[0]
V1_map = Res_map[1]
T2_map = Res_map[3]
V1_map /= 7 # Para re-escalar los resultados a una escala realista
# Datos reales
Data_r = OutData(nmrec=7*1440) # Los datos sí se leen completos
C1_r = Data_r[0]
V1_r = Data_r[1]
T2_r = Data_r[2]
V1_r /= 7 # Para re-escalar los resultados a una escala realista
## Gráficas soluciones
t = arange(1,7*1440+1)
# Media
plot(t, C1_r)
plot(t[2*1440:], C1_mn)
xlabel('mt')
ylabel(r'$\frac{mg}{m^3}$')
title(r'$C_1$')
figure()
plot(t, V1_r)
plot(t[2*1440:], V1_mn)
xlabel('mt')
ylabel(r'$Pa$')
title(r'$V_1$')
figure()
plot(t, T2_r)
plot(t[2*1440:], T2_mn)
xlabel('mt')
ylabel(r'$C$')
title(r'$T_2$')
# Mediana
figure()
plot(t, C1_r)
plot(t[2*1440:], C1_md)
xlabel('mt')
ylabel(r'$\frac{mg}{m^3}$')
title(r'$C_1$')
figure()
plot(t, V1_r)
plot(t[2*1440:], V1_md)
xlabel('mt')
ylabel(r'$Pa$')
title(r'$V_1$')
figure()
plot(t, T2_r)
plot(t[2*1440:], T2_md)
xlabel('mt')
ylabel(r'$C$')
title(r'$T_2$')
# Map
figure()
plot(t, C1_r)
plot(t[2*1440:], C1_map)
xlabel('mt')
ylabel(r'$\frac{mg}{m^3}$')
title(r'$C_1$')
figure()
plot(t, V1_r)
plot(t[2*1440:], V1_map)
xlabel('mt')
ylabel(r'$Pa$')
title(r'$V_1$')
figure()
plot(t, T2_r)
plot(t[2*1440:], T2_map)
xlabel('mt')
ylabel(r'$C$')
title(r'$T_2$')

## Incertidumbre en las soluciones
ns = int(len(alpha1)/10) # número de soluciones
C1_Sol = zeros( ( 5*1440, ns ) ) 
V1_Sol = zeros( ( 5*1440, ns ) ) 
T2_Sol = zeros( ( 5*1440, ns ) ) 
for i in range(ns):
    theta_t = array([ alpha1[i*10], phi2[i*10], psi2[i*10] ])
    Soluc = Sol_Climate(theta=theta_t, nmrec=5*1440, Inf=False)
    C1_Sol[:,i] = Soluc[0]
    V1_Sol[:,i] = Soluc[1]
    T2_Sol[:,i] = Soluc[3]

V1_Sol /= 7 # Para re-escalar los resultados a una escala realista

savetxt('C1_Sol_1k', C1_Sol)
savetxt('V1_Sol_1k', V1_Sol)
savetxt('T2_Sol_1k', T2_Sol)

## Cuantiles de la solución
# C1
q10_C1 = zeros(5*1440)
q25_C1 = zeros(5*1440)
q50_C1 = zeros(5*1440)
q75_C1 = zeros(5*1440)
q90_C1 = zeros(5*1440)
# V1
q10_V1 = zeros(5*1440)
q25_V1 = zeros(5*1440)
q50_V1 = zeros(5*1440)
q75_V1 = zeros(5*1440)
q90_V1 = zeros(5*1440)
# T2
q10_T2 = zeros(5*1440)
q25_T2 = zeros(5*1440)
q50_T2 = zeros(5*1440)
q75_T2 = zeros(5*1440)
q90_T2 = zeros(5*1440)

for i in range(5*1440):
    # C1
    q10_C1[i] = quantile(C1_Sol[i,:], 0.1)  # se rompió aquí
    q25_C1[i] = quantile(C1_Sol[i,:], 0.25)  
    q50_C1[i] = quantile(C1_Sol[i,:], 0.5)  
    q75_C1[i] = quantile(C1_Sol[i,:], 0.75)  
    q90_C1[i] = quantile(C1_Sol[i,:], 0.9)  
    # V1
    q10_V1[i] = quantile(V1_Sol[i,:], 0.1)  
    q25_V1[i] = quantile(V1_Sol[i,:], 0.25)  
    q50_V1[i] = quantile(V1_Sol[i,:], 0.5)  
    q75_V1[i] = quantile(V1_Sol[i,:], 0.75)  
    q90_V1[i] = quantile(V1_Sol[i,:], 0.9)  
    # C1
    q10_T2[i] = quantile(T2_Sol[i,:], 0.1)  
    q25_T2[i] = quantile(T2_Sol[i,:], 0.25)  
    q50_T2[i] = quantile(T2_Sol[i,:], 0.5)  
    q75_T2[i] = quantile(T2_Sol[i,:], 0.75)  
    q90_T2[i] = quantile(T2_Sol[i,:], 0.9)  

## Gráfica de la solución con cuantiles
# C1
plot(t[2*1440:], q10_C1, color='black', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q25_C1, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q50_C1, color='orange', linewidth=0.7)    
plot(t[2*1440:], q75_C1, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q90_C1, color='black', linestyle=':', linewidth=0.5)    
plot(t, C1_r, linewidth=0.7)
xlabel('mt')
ylabel(r'$\frac{mg}{m^3}$')
title(r'$C_1$')
# V1
plot(t[2*1440:], q10_V1, color='black', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q25_V1, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q50_V1, color='orange', linewidth=0.7)    
plot(t[2*1440:], q75_V1, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q90_V1, color='black', linestyle=':', linewidth=0.5)    
plot(t, V1_r, linewidth=0.7)
xlabel('mt')
ylabel(r'$Pa$')
title(r'$V_1$')
# T2
plot(t[2*1440:], q10_T2, color='black', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q25_T2, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q50_T2, color='orange', linewidth=0.7)    
plot(t[2*1440:], q75_T2, color='red', linestyle=':', linewidth=0.5)    
plot(t[2*1440:], q90_T2, color='black', linestyle=':', linewidth=0.5)    
plot(t, T2_r, linewidth=0.7)
xlabel('mt')
ylabel(r'$C$')
title(r'$T_2$')
