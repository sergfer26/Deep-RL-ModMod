#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:55:46 2020

@author: jdmolinam
"""

##################################################################
##################################################################
################# Inferencia - Módulo Producción ##################
##################################################################
##################################################################


###########################
####### Módulos  ##########
###########################
from Solver_Pdn import Sol_Pdn, OutData_Pdn
from scipy.stats import gamma
import pytwalk
from numpy import sqrt, log, array, mean, median, argmin, arange, savetxt, zeros, loadtxt, quantile, random
from matplotlib.pyplot import hist, figure, xlabel, ylabel, axvline, plot, title

##################################################################
################ Ejecución del MCMC ##############################
##################################################################

## Tiempo de inferencia 
di = 115 # días inferencia

##################################################################
################ Ejecución del MCMC ##############################
##################################################################

### Soporte del parámetro ###
def ProductionSupp(theta):
    """
    Se evalúa que un candidato para la a posteriori de theta
    efectivamente esté en el soporte del parámetro.
    theta = [nu, a, b, sigma_F].
    """
    rt = True
    rt &= (0 < theta[0] < 1)
    rt &= (theta[1] > 0)
    rt &= (0 < theta[2] < 1)
    rt &= (theta[3] > 0)
    return rt

### Valores iniciales ###
def ProductionInit():
    """
    Genera valores iniales para theta 
    a partir de su distribución a priori.
    theta = [nu, a, b, sigma_F].
    """
    nu = random.random()
    a = gamma.rvs(a=0.01, scale=1/3.03e-3)
    b = random.random()
    sigma_F = gamma.rvs(a=16, scale=1/16)
    return array( [nu, a, b, sigma_F] )

### Función energía ###
def ProductionU(theta):
    """
    Esta es la función energía:
    U = - log [ f (theta | Datos) ].
    theta = [nu, a, b, sigma_F].
    """
    ## Se soluciona el sistema para nmrec minutos
    k = di
    Res = Sol_Pdn(theta=theta)
    m = Res[0]
    n = Res[1]
    h = Res[2]
    ## Cálculo de la energía
    U = 0.99*log(theta[1]) + (k - 15)*log(theta[3]) + 3.03e-3*theta[1] + 16*theta[3] \
    + sum( (16*n)/(m + 1e-6) ) + 0.5*sum( ( (h - 380*n)/( sqrt(n + 1e-6)*theta[3] ) )**2 ) \
    + sum( 16*log(m + 1e-6) - 14.5*log(n + 1e-6) )
    return U

## Ejecución del MCMC ###
Init0 = array([0.8, 3.5, 0.3, 1.2]) 
Init1 = array([0.6, 3.1, 0.2, 0.8]) 
Production = pytwalk.pytwalk( n=4, U=ProductionU, Supp=ProductionSupp)
Production.Run( T=1200, x0=Init0, xp0=Init1)

# Análsis resultados
#Climate.Ana() 
## Extracción de resultados
param = Production.Output # Las primeras columnas son los paramétros y la última los valores de la energía
#Production.Save('param_10k', start=1000) # se esta haciendo un burn-in de 1000
#param = loadtxt('param_10k')
start = 200
nu = param[start: , 0]
a = param[start: , 1]
b = param[start: , 2]
sigma_F = param[start: , 3]
U = param[start: , 4]

## Se guardan los resultados  -- Los datos pueden cargarse con la funcion loadtxt de numpy
savetxt('nu_1k', nu)
savetxt('a_1k', a)
savetxt('b_1k', b)
savetxt('sigmaF_1k', sigma_F)
savetxt('U_1k', U)

"""
## Estimadores a posteriori
# media a posteriori
nu_mn = mean(nu)
a_mn = mean(a)
b_mn = mean(b)
sigmaF_mn = mean(sigma_F)
theta_mn = array([nu_mn, a_mn, b_mn, sigmaF_mn])
# mediana a posteriori
nu_md = median(nu)
a_md = median(a)
b_md = median(b)
sigmaF_md = median(sigma_F)
theta_md = array([nu_md, a_md, b_md, sigmaF_md])
# MAP
idMax = argmin(U) # El mínimo de la energía es el máximo de la log posterior, pues U = - log (post)
nu_map = nu[idMax]
a_map = a[idMax]
b_map = b[idMax]
sigmaF_map = sigma_F[idMax]
theta_map = array([nu_map, a_map, b_map, sigmaF_map])

### Histogramas
hist(nu, density=True)
axvline(x=0.7, color='red')
xlabel(r'$\nu$')
ylabel("Density")
figure()
hist(a, density=True)
axvline(x=3.3, color='red')
xlabel(r'$a$')
ylabel("Density")
figure()
hist(b, density=True)
axvline(x=0.25, color='red')
xlabel(r'$b$')
ylabel("Density")
figure()
hist(sigma_F, density=True)
axvline(x=1., color='red')
xlabel(r'$\sigma_F$')
ylabel("Density")

## Soluciones con los estimadores posteriores
# Solución con la media
Res_mn = Sol_Pdn(theta=theta_mn) # No se incluye el tiempo de inferencia y se usa la base de predicción
NF_mn = Res_mn[3]
H_mn = Res_mn[4]
# Solución con la media
Res_md = Sol_Pdn(theta=theta_md) # No se incluye el tiempo de inferencia y se usa la base de predicción
NF_md = Res_md[3]
H_md = Res_md[4]
# Solución con la media
Res_map = Sol_Pdn(theta=theta_map) # No se incluye el tiempo de inferencia y se usa la base de predicción
NF_map = Res_map[3]
H_map = Res_map[4]
# Datos reales
Data = OutData_Pdn()
NF_H = Data[0]
H_H = Data[1]

## Gráficas soluciones
t = arange(1,di+1)
# Media
plot(t, NF_H)
plot(t, NF_mn)
xlabel('Day')
ylabel('Number fruits')
title(r'$NF$')
figure()
plot(t, H_H)
plot(t, H_mn)
xlabel('Day')
ylabel(r'$g$')
title(r'$H$')
# Mediana
plot(t, NF_H)
plot(t, NF_md)
xlabel('Day')
ylabel('Number fruits')
title(r'$NF$')
figure()
plot(t, H_H)
plot(t, H_md)
xlabel('Day')
ylabel(r'$g$')
title(r'$H$')
# Media
plot(t, NF_H)
plot(t, NF_map)
xlabel('Day')
ylabel('Number fruits')
title(r'$NF$')
figure()
plot(t, H_H)
plot(t, H_map)
xlabel('Day')
ylabel(r'$g$')
title(r'$H$')
"""

## Incertidumbre en las soluciones
ns = int(len(nu)/10) # número de soluciones
NF_Sol = zeros( ( di, ns ) ) 
H_Sol = zeros( ( di, ns ) ) 
for i in range(ns):
    theta_t = array([ nu[i*10], a[i*10], b[i*10], sigma_F[i*10] ])
    Soluc = Sol_Pdn(theta=theta_t)
    NF_Sol[:,i] = Soluc[3]
    H_Sol[:,i] = Soluc[4]
    
savetxt('NF_Sol_1k', NF_Sol)
savetxt('H_Sol_1k', H_Sol)

"""
## Cuantiles de la solución
# NF
q10_NF = zeros(di)
q25_NF = zeros(di)
q50_NF = zeros(di)
q75_NF = zeros(di)
q90_NF = zeros(di)
# C1
q10_H = zeros(di)
q25_H = zeros(di)
q50_H = zeros(di)
q75_H = zeros(di)
q90_H = zeros(di)

for i in range(di):
    # NF
    q10_NF[i] = quantile(NF_Sol[i,:], 0.1)  
    q25_NF[i] = quantile(NF_Sol[i,:], 0.25)  
    q50_NF[i] = quantile(NF_Sol[i,:], 0.5)  
    q75_NF[i] = quantile(NF_Sol[i,:], 0.75)  
    q90_NF[i] = quantile(NF_Sol[i,:], 0.9)  
    # H
    q10_H[i] = quantile(H_Sol[i,:], 0.1)  
    q25_H[i] = quantile(H_Sol[i,:], 0.25)  
    q50_H[i] = quantile(H_Sol[i,:], 0.5)  
    q75_H[i] = quantile(H_Sol[i,:], 0.75)  
    q90_H[i] = quantile(H_Sol[i,:], 0.9)  

## Gráfica de la solución con cuantiles
# NF
plot(t, q10_NF, color='black', linestyle=':', linewidth=0.5)    
plot(t, q25_NF, color='red', linestyle=':', linewidth=0.5)    
plot(t, q50_NF, color='orange', linewidth=0.7)    
plot(t, q75_NF, color='red', linestyle=':', linewidth=0.5)    
plot(t, q90_NF, color='black', linestyle=':', linewidth=0.5)    
plot(t, NF_H, linewidth=0.7)
xlabel('Day')
ylabel('Number fruits')
title(r'$NF$')
figure()
# H
plot(t, q10_H, color='black', linestyle=':', linewidth=0.5)    
plot(t, q25_H, color='red', linestyle=':', linewidth=0.5)    
plot(t, q50_H, color='orange', linewidth=0.7)    
plot(t, q75_H, color='red', linestyle=':', linewidth=0.5)    
plot(t, q90_H, color='black', linestyle=':', linewidth=0.5)    
plot(t, H_H, linewidth=0.7)
xlabel('Day')
ylabel(r'$g$')
title(r'$H$')
"""