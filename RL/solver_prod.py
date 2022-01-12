#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:49:18 2020

@author: jdmolinam
"""
# T2 - ©, 339
# C1 - 331
# V1 - 487

# masa seca 735


#Solver - Módulo produccion


###########################
####### Módulos  ##########
###########################
import numpy as np
import copy
from numpy import exp, floor, clip, arange, append
from sympy import symbols
from ModMod import Module, StateRHS, Director, ReadModule
from sympy.utilities.iterables import bracelets


#Simbolos

#Photosynthesis model
s, mol_CO2, mol_air, mol_phot, m, d, C, g, mol_O2, pa, ppm = symbols('s mol_CO2 mol_air mol_phot m d C g mol_O2 pa ppm')
mu_mol_CO2 = 1e-6 * mol_CO2
mu_mol_phot = 1e-6 * mol_phot
mu_mol_O2 = 1e-6 * mol_O2
mg = 1e-3*g

## Growth model
n_f, n_p, MJ = symbols('n_f n_p MJ') # number of fruits, number of plants

#Funciones auxiliares 

#Fotosíntesis
from plant.photosynthesis_model.functions import *

#Modelo de crecimiento
from production_model.functions import * 

#RHS del CO2 intracelular   
from plant.photosynthesis_model.ci_rhs import Ci_rhs

#Módulo de crecimiento para una planta
from production_model.q_rhs import Q_rhs


from plant.plant_director import PlantDirector
#Director principal del invernadero


class GreenHouse(Director):
    def __init__(self):
        super().__init__(t0=0.0, time_unit="", Vars={}, Modules={})
        beta_list = [0.99, 0.95] # Only 2 plants are simulated, assuming this is approximately one m**2 -> radiación de dos plantas (distinto)
        self.PlantList = []
        for p, beta in enumerate(beta_list):
            ### Make and instance of a Plant
            Dir = PlantDirector(beta=beta)
            ### Merge all ***global*** vars from plant
            self.MergeVars( [ Dir ], call=__name__)
            ### Add the corresponding time unit, most be the same in both
            self.AddTimeUnit(Dir.symb_time_unit)
            #Model.CheckSymbTimeUnits, all repeated instances of the Plant Director-Module 
            ### Add Plant directly, Dir.sch has been already defined
            self.AddDirectorAsModule( "Plant%d" % p, Dir)
            self.PlantList +=["Plant%d" % p]

        self.sch = self.PlantList.copy()
        ## Add global variables
        self.AddVarLocal( typ='State', varid='H', prn=r'$H_k$',\
           desc="Accumulated weight of all harvested fruits.", units= g, val=0) # peso total de los pepinos
        self.AddVarLocal( typ='State', varid='NF', prn=r'$N_k$',\
           desc="Accumulated  number of fruits harvested", units= n_f, val=0)
        self.AddVarLocal( typ='State', varid='h', prn=r'$h_k$',\
           desc="Weight of all harvested fruits.", units= g, val=0)
        self.AddVarLocal( typ='State', varid='n', prn=r'$n_k$',\
           desc="Total  number of fruits harvested", units= n_f, val=0)

    def Scheduler( self, t1, sch):
        
        """Advance the modules to time t1. sch is a list of modules id's to run
           its Advance method to time t1.
           
           Advance is the same interface, either if single module or list of modules.
        """
        
        for mod in sch:
            if self.Modules[mod].Advance(t1) != 1:
                print("Director: Error in Advancing Module '%s' from time %f to time %f" % ( mod, self.t, t1))
        self.t = t1
        
        ### Update Total weight and total number of fruits
        #t_w_current = 0.0
        t_w_hist = 0.0 #peso cultivado de las plantas acumulado
        t_n_f = 0 # número de frutos cosechados acumulado 
        t_w_k = 0.0 # peso frutos cosechados por día
        t_n_k = 0 # numéro de frutos cosechados por día
        for _, plant in enumerate(self.PlantList):
            #t_w_current += Model.Modules[plant].Modules['Plant'].V('Q')
            t_w_hist += self.Modules[plant].Modules['Plant'].V('Q_h')
            t_n_f += self.Modules[plant].Modules['Plant'].n_fruits_h 
            t_w_k += self.Modules[plant].Modules['Plant'].V('h_k')
            t_n_k += self.Modules[plant].Modules['Plant'].V('n_k')
        self.V_Set( 'H', t_w_hist)
        self.V_Set( 'NF', t_n_f)
        self.V_Set( 'h', t_w_k)
        self.V_Set( 'n', t_n_k)

    def reset(self):
        self.Vars['H'].val = 0
        self.Vars['NF'].val = 0
        self.Vars['h'].val = 0
        self.Vars['n'].val = 0
        for _, plant in enumerate(self.PlantList):
            self.Modules[plant].Modules['Plant'].veget = [0.0 , 0.0] ## characteristics for vegetative part: Weight and growth potential 
            self.Modules[plant].Modules['Plant'].fruits = [] # No fruits
            self.Modules[plant].Modules['Plant'].n_fruits = 0 ## Current number of fruits
            self.Modules[plant].Modules['Plant'].n_fruits_h = 0 ## total number of fruits harvested
            self.Modules[plant].Modules['Plant'].new_fruit = 0  ## Cummulative number of fruits
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['A'].val = 0
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['Q'].val = 0
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['n_k'].val = 0
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['h_k'].val = 0
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['Q_h'].val = 0
            self.Modules[plant].Modules['Plant'].StateRHSs['Q'].mod.D.Vars['Y_sum'].val = 0
            #self.Modules[plant].Modules['Plant'].StateRHSs['Ci'].__init__()


    def update_state(self, C1, T, PAR, RH):
        self.Vars['C1'].val = C1 
        self.Vars['T'].val = T
        self.Vars['PAR'].val = PAR
        self.Vars['RH'].val = RH
        