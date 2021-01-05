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

##################################################################
##################################################################
################# Solver - Módulo producción #####################
##################################################################
##################################################################

###########################
####### Módulos  ##########
###########################
import numpy as np
from numpy import exp, floor, clip, arange, append
from sympy import symbols
from ModMod import Module, StateRHS, Director, ReadModule
from solver_climate import Dir as DirClim

###########################
####### Símbolos  #########
###########################
## Photosynthesis model
s, mol_CO2, mol_air, mol_phot, m, d, C, g, mol_O2, pa, ppm = symbols('s mol_CO2 mol_air mol_phot m d C g mol_O2 pa ppm')
mu_mol_CO2 = 1e-6 * mol_CO2
mu_mol_phot = 1e-6 * mol_phot
mu_mol_O2 = 1e-6 * mol_O2
mg = 1e-3*g
## Growth model
n_f, n_p, MJ = symbols('n_f n_p MJ') # number of fruits, number of plants


###########################
###### Funciones ##########
###########################

##### Fotosíntesis #######
## Funciones auxiliares ##
def V_cmax (T_f, V_cmax25, Q10_Vcmax, k_T, k_d):
    pow1 = (T_f - 25*k_T)/(10*k_T) 
    pow2 =  0.128 * (T_f - 40*k_T) / (1*k_T) 
    return (V_cmax25*k_d) * Q10_Vcmax**( pow1 ) / ( 1 + exp(pow2) )  

def Gamma_st (T_f): # Esta función la estoy calculando como lo hace Aarón
    return 150 * exp( 26.355 - ( 65.33 / ( 0.008314 * (T_f + 273.15) ) ) ) # La temperatura se está pasando de °C a Kelvins

def tau (T_f, tau_25, Q10_tau, k_T):
    return tau_25 * Q10_tau**( (T_f - 25*k_T)/(10*k_T) )

def K_C (T_f, K_C25, Q10_KC, k_T):
    return K_C25 * Q10_KC**( (T_f - 25*k_T)/(10*k_T) )
    
def K_O (T_f, K_O25, Q10_KO, k_T):
    return K_O25 * Q10_KO**( (T_f - 25*k_T)/(10*k_T) )

def I_2 (I, f, ab):
    return  I * ab*(1 - f)  / 2

def J (I_2, J_max, theta, k_d):
    return ( (I_2*k_d) + (J_max*k_d) + ( ( (I_2 + J_max)*k_d )**2 -4*theta*I_2*k_d*J_max*k_d )**(0.5) ) / (2*theta)

## Factores limitantes en la producción de asimilados ##    
def A_R (O_a, tau, C_i, V_cmax, Gamma_st, K_C, K_O, phi): 
    """
    Asimilación por Rubisco
    """
    C_i1 = C_i*(28.96/44) # El CO2 está pasando de ppm a (mu_mol_CO2/mol_air)
    return ( 1 - ( O_a / (phi*tau*C_i1) ) ) * V_cmax * (C_i1 - Gamma_st) / ( K_C *(1 + (O_a/K_O) ) + C_i1 )

def A_f (C_i, Gamma_st, J, k_JV): 
    """
    Asimilación por radiación PAR
    """
    C_i1 = C_i*(28.96/44) # El CO2 está pasando de ppm a (mu_mol_CO2/mol_air)
    return ( (C_i1 - Gamma_st)*J / ( 4*C_i1 + 8*Gamma_st) )*k_JV

def A_acum(V_cmax):
    """
    Asimilación por acumulación de carbohídratos
    """
    return V_cmax/2

def R_d (V_cmax):
    """
    Asimilados empleados en el mantenimiento de la planta
    """
    return 0.015*V_cmax

# Tasa de asimilación
def A (A_R, A_f, A_acum, R_d, fc):
    return fc * ( min( A_R, A_f, A_acum ) - R_d )

#### Resistencia Estomática ####
    
def r_s (r_m, f_R, f_C, f_V, k_d):
    """
    En esta función se cálcula la resistencia estomática
    la cual depende r_m, el mínimo valor de resistenccia estomática,
    de f_R, f_c y f_V que son los factores de resistencia debidas a
    la radiación, el CO2 y la presión de vapor respectivamente. 
    """
    return (r_m/k_d) * f_R * f_C * f_V  # f_R, f_C y f_V son adimensionales

def f_R (I, C_ev1, C_ev2):
    """
    Factor de resistencia debida a la radiación global I
    """
    return (I + C_ev1) / (I + C_ev2)

def f_C (C_ev3, C1, k_fc):
    """
    Factor de resistencia debida al CO2 
    """
    return 1 + C_ev3*( (C1 - 200*k_fc)**2 )

def C_ev3 (C_ev3n, C_ev3d, Sr):
    return C_ev3n*(1 - Sr) + C_ev3d*Sr

def Sr (I, S, Rs):
    return 1 / ( 1 + exp( S*(I - Rs) ) )

def f_V (C_ev4, VPD):
    return 1 + C_ev4*( VPD**2 )

def C_ev4 (C_ev4n, C_ev4d, Sr):
    return C_ev4n*(1 -Sr) + C_ev4d*Sr

def VPD(V_sa, RH): # Vapour pressure deficit
    return V_sa*(1 - RH/100)

def V_sa (T):
    """
    Esta función cálcula la presión de vapor de saturación,
    basandose en la ecuación de Arden Buck. T es la temperatura del aire en °C.
    Ver: https://en.wikipedia.org/wiki/Arden_Buck_equation
    """
    return 0.61121 * exp( ( 18.678 - (T/234.5) ) * ( T /(257.14 + T) ) )


#### Cálculo del CO2 intracelular ####
### Flujo de absorción del CO2 ###
def gTC (k, Rb, Rs, k_d):
    """
    Esta función calcula la conductancia total de CO2
    para una determinada capa del dosel
    """
    gs = (1 / Rs) # Conductancia estomática
    gb = (1 / Rb)*k_d # Conductancia estomática de la capa límite del dosel
    gtc = ( (1+k)*(1.6/gs) + (1.37/gb) )**-1 + k*( ( (1+k)*(1.6/gs) + k*(1.37/gb) )**-1 )
    return gtc

def Ca (gtc, C, Ci):
    """
    Esta función calcula el CO2 absorbido (mg / m**2 * d) 
    por una determinada capa del dosel
    """
    C1 = C*(44/28.96) # El CO2 pasa de (mu_mol_CO2/mol_air) a ppm
    return gtc*( (C1 - Ci)*0.554 ) # El CO2 se está pasando de ppm a mg/m**2


#### Modelo de crecimiento ####
"""
*******  Note, the original TF is quite unsensitive to PA_mean ******
def TF( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    ### ORIGINAL Floration rate. DOES NOT DEPEND MUCH ON PA_MEAN!!
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-(1*k1_TF+PA_mean)/(2*k1_TF)))*k3_TF * Dt
"""

def TF_tmp( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    """Floration rate. TEMPORARY ... k1_TF = 150 the original below look wrong"""
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-PA_mean/k1_TF))*k3_TF * Dt

def TF( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    ### ORIGINAL Floration rate. DOES NOT DEPEND MUCH ON PA_MEAN!!
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-(1*k1_TF+PA_mean)/(2*k1_TF)))*k3_TF * Dt


def f( k1_TF, k2_TF, PA_mean, T_mean, Dt):
    return -(1*k1_TF+PA_mean)/(2*k1_TF)

def Y_pot(k2_TF, C_t, B, D, M, X, T_mean):
    """Growth potential of each fruit."""
    return (T_mean - 10*k2_TF) *\
        B * M * exp(B * ( X - C_t))/(1 + D * exp(B * (X - C_t)))**( 1 + 1/D)

def Y_pot_veg(k2_TF, a, b, T_mean):
    """Growth potencial of the vegetative part"""
    return a*k2_TF + b*T_mean

def t_wg( dw_ef, A, f_wg):
    """Growth rate."""
    return f_wg * A / dw_ef

def f_wg( Y_pot, Y_sum):
    """Sink stregth, without kmj."""
    return Y_pot / Y_sum  ### No units
##################################################################
######### Función que soluciona el módulo de producción ##########
################################################################## 
S_mean = [432, 14, 20, 20]
nmrec=115
theta = np.array([0.7, 3.3, 0.25])
"""
Se calcula la producción del invernadero en función 
de los parámetros contenidos en el vector theta.
theta = [ nu, a, b ].
nmrec es el número de resultados de cada variable que serán reportados.
La función retorna resultados en este orden: ( NF, H ).
"""

C1M, _, _, T2M  = S_mean
#################################################################
############ RHS modelo de crecimiento ##########################
#################################################################    
class Q_rhs(StateRHS):
    """
    Q is the weight of all fruits for plant
    """
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        ### uses the super class __init__
        super().__init__()
        
        ### Define variables here.  Each fruit will have repeated variables.
        ### Later some will be shared and the Local variable swill be exclusive
        ### of each fruit.
        
        nrec = nmrec # Number of outputs that will be record

        self.SetSymbTimeUnits(d) # days

        ### State variables coming from the climate model
        self.AddVar( typ='State', varid='T', prn=r'$T$',\
            desc="Greenhouse air temperature", units= C , val=T2M, rec=nrec) #T2 nos interesa
        
        self.AddVar( typ='State', varid='PAR', prn=r'$PAR$',\
            desc="PAR radiation", units=mu_mol_phot * (m**-2) * d**-1 , val=300.00, rec=nrec)
        
        ### Local variables, separate for each plant
        self.AddVarLocal( typ='State', varid='A', prn=r'$A$',\
           desc="Assimilation rate", units= g * (m**-2), val=0, rec=nrec)
        
        self.AddVarLocal( typ='StatePartial', varid='Q', prn=r'$Q$',\
           desc="Weight of all fruits for plant", units= g, val=0.0)
        
        self.AddVarLocal( typ='StatePartial', varid='n_k', prn=r'$n_k$',\
           desc="Number of fruits harvested for plant", units= n_f, val=0)

        self.AddVarLocal( typ='StatePartial', varid='h_k', prn=r'$h_k$',\
           desc="Weight of all harvested fruits for plant", units= g, val=0.0)

        self.AddVarLocal( typ='StatePartial', varid='Q_h', prn=r'$H$',\
           desc="Accumulated weight of all harvested fruits for plant", units= g, val=0.0)

        self.AddVarLocal( typ='StatePartial', varid='Y_sum', prn=r'$Y_{sum}$',\
           desc="Sum of all potentail growths", units= g/d**2, val=0.0)


        ### Canstants, shared by all plants.  Shared Cnts cannot be local
        self.AddVar( typ='Cnts', varid='k1_TF', prn=r'$k1_TF$',\
           desc="Aux in function TF", units= MJ * m**-2 * d**-1, val=300.0)

        self.AddVar( typ='Cnts', varid='k2_TF', prn=r'$k2_TF$',\
           desc="Aux in function TF", units= C * d**-1, val=1.0)

        self.AddVarLocal( typ='Cnts', varid='k3_TF', prn=r'$k3_TF$',\
           desc="Aux in function TF", units= n_f * C**-1, val=1.0)

        self.AddVarLocal( typ='Cnts', varid='dw_ef', prn=r'$dw_{efficacy}$',\
           desc="Constant in t_wg for fruits", units= 1, val=1.3)
        
        self.AddVarLocal( typ='Cnts', varid='dw_ef_veg', prn=r'$dw_{efficacy}$',\
           desc="Constant in t_wg for vegetative part", units= 1, val=1.15)

        self.AddVarLocal( typ='Cnts', varid='a_ef', prn=r'$a_{efficacy}$',\
           desc="Matching constant in remaining assimilates", units= 1/m**2, val=1.0)

        self.AddVarLocal( typ='Cnts', varid='C_t', prn=r'$C_t$',\
           desc="Constant in Y_pot", units= C * d, val=131.0)

        self.AddVarLocal( typ='Cnts', varid='B', prn=r'$B$',\
           desc="Constant in Y_pot", units= (C * d)**-1, val=0.017)

        self.AddVarLocal( typ='Cnts', varid='D', prn=r'$D$',\
           desc="Constant in Y_pot", units= 1, val=0.011)

        self.AddVarLocal( typ='Cnts', varid='M', prn=r'$M$',\
           desc="Constant in Y_pot", units= g, val=60.7)
        
        self.AddVarLocal( typ='Cnts', varid='a', prn=r'$a$',\
           desc="Constant in Y_pot_veg", units= 1, val=theta[1])
        
        self.AddVarLocal( typ='Cnts', varid='b', prn=r'$b$',\
           desc="Constant in Y_pot_veg", units= 1, val=theta[2])

    def RHS( self, Dt):
        """RHS( Dt ) = 
           
        ************* IN ASSIGMENT RHSs WE DON'T NEED TO CALL STATE VARS WITH self.Vk ******************
        """
        ### The assigment is the total weight of the fuits
        return self.V('Q')


#################################################################
############ RHS del CO2 intracelular ###########################
#################################################################    

class Ci_rhs(StateRHS):
    """
    Ci es el CO2 intracelular 
    """
    def __init__( self ):
        ### uses the super class __init__
        super().__init__()
        nrec = nmrec # Number of outputs that will be record
        self.SetSymbTimeUnits(d) # días
        ### Add variables ###
        ## State variables
        self.AddVarLocal( typ='State', varid='Ci', prn=r'$C_i$',\
                    desc="Intracellular CO2", units= ppm , val=410, rec=nrec)
        
        self.AddVarLocal( typ='State', varid='A', prn=r'$A$',\
           desc="Assimilation rate", units= g * (m**-2), val=0, rec=nrec)
        
        ## Inputs
        self.AddVar( typ='State', varid='C1', prn=r'$C_1$',\
           desc="CO2 concentration in the greenhouse air", \
           units= mu_mol_CO2 * mol_air**-1, val=C1M) # C1 nos interesa
        
        self.AddVar( typ='State', varid='RH', prn=r'$RH$',\
           desc="Relative humidity percentage in the greenhouse air", \
           units=1, val=50)
        
        self.AddVar( typ='State', varid='T', prn=r'$T$',desc="Greenhouse air temperature", units= C , val=T2M, rec=nrec) # T2 (nos interesa)
        
        self.AddVar( typ='State', varid='PAR', prn=r'$PAR$',\
            desc="PAR radiation", units=mu_mol_phot * (m**-2) * d**-1 , val=300.00, rec=nrec)
        
        ## Canstants
        ### Stomatal Resistance Calculation
        self.AddVar( typ='Cnts', varid='k_Ag', \
           desc="Constant for units transformation", \
           units= m**3 * g**-1 * s**-1 * mu_mol_CO2 * mol_air**-1, val=1)
        
        self.AddVar( typ='Cnts', varid='r_m', \
           desc="minimal stomatal resistance", \
           units= s * m**-1, val=100)
        
        self.AddVar( typ='Cnts', varid='C_ev1', \
           desc="Constant in the formula of f_R", \
           units= mu_mol_phot * (m**-2) * d**-1, val=4.3)
        
        self.AddVar( typ='Cnts', varid='C_ev2', \
           desc="Constant in the formula of f_R", \
           units= mu_mol_phot * (m**-2) * d**-1 , val=0.54)
        
        self.AddVar( typ='Cnts', varid='k_fc', \
           desc="Constant for units completation", \
           units= mu_mol_CO2 * mol_air**-1, val=1)
        
        self.AddVar( typ='Cnts', varid='C_ev3d', \
           desc="Constant in the formula of f_C", \
           units= mol_air * mu_mol_CO2**-1, val=6.1e-7)
        
        self.AddVar( typ='Cnts', varid='C_ev3n', \
           desc="Constant in the formula of f_C", \
           units= mol_air * mu_mol_CO2**-1, val=1.1e-11)
        
        self.AddVar( typ='Cnts', varid='S', \
           desc="Constant in the formula of Sr", \
           units= m**2 * d * mu_mol_phot**-1, val=-1)
        
        self.AddVar( typ='Cnts', varid='Rs', \
           desc="Radiation setpoint to switch day and night", \
           units= mu_mol_phot * (m**-2) * d**-1, val=5)
        
        self.AddVar( typ='Cnts', varid='C_ev4d', \
           desc="Constant in the formula of f_C", \
           units= pa**-1, val=4.3e-6)
        
        self.AddVar( typ='Cnts', varid='C_ev4n', \
           desc="Constant in the formula of f_C", \
           units= pa**-1, val=5.2e-6)
        
        ## CO2 absorption
        self.AddVar( typ='Cnts', varid='ks', \
           desc="Stomatal ratio", \
           units= 1, val=0.5)
        
        self.AddVar( typ='Cnts', varid='Rb', \
           desc="Stomatal resistance of the canopy boundary layer", \
           units= s * m**-1, val=711)
        
        ## Assimilates
        self.AddVar( typ='Cnts', varid='k_d', \
           desc="factor to transform s**-1 into d**-1", units=1, val=1)
        
        self.AddVar( typ='Cnts', varid='k_T', \
           desc="Auxiliary constant to add temperature units", units= C, val=1.0)
        
        self.AddVar( typ='Cnts', varid='k_JV', \
           desc="Auxiliary constant which transforms the units of the electron transport rate, J to those of the maximum Rubisco rate, V_cmax", \
           units= mu_mol_CO2 * mu_mol_phot**-1, val=1.0)
        
        self.AddVar( typ='Cnts', varid='fc', \
           desc="Factor to transform mu-mols_CO2/sec to grms_CH20/day", \
           units= g * d * mu_mol_CO2**-1 , val=3.418181e-1) # 7.891414141414142e-6
        
        self.AddVar( typ='Cnts', varid='phi', \
           desc="Ratio of oxigenation to carboxylation rates", \
           units= mu_mol_O2 * mu_mol_CO2**-1, val=2)
        
        self.AddVar( typ='Cnts', varid='O_a', \
           desc="O2 concentration in the enviroment", \
           units= mu_mol_O2 * mol_air**-1, val=210000)
        
        self.AddVar( typ='Cnts', varid='V_cmax25', \
           desc="Maximum Rubisco Rate, per unit area", \
           units= mu_mol_CO2 * (m**-2) * d**-1, val=200)
        
        self.AddVar( typ='Cnts', varid='Q10_Vcmax', \
           desc="Temperatura response of Vcmax", \
           units=1, val=2.4)
        
        self.AddVar( typ='Cnts', varid='K_C25', \
           desc="Michaelis-Menten for CO2", \
           units= mu_mol_CO2 * mol_air**-1 , val=300)
        
        self.AddVar( typ='Cnts', varid='Q10_KC', \
           desc="Temperatura response of Michaelis-Menten for CO2", \
           units=1, val=2.1)
        
        self.AddVar( typ='Cnts', varid='K_O25', \
           desc="Michaelis-Menten for O2", \
           units= mu_mol_O2 * mol_air**-1 , val=3e5)
        
        self.AddVar( typ='Cnts', varid='Q10_KO', \
           desc="Temperatura response of Michaelis-Menten for O2", \
           units=1, val=1.2) 
        
        self.AddVar( typ='Cnts', varid='tau_25', \
           desc="Specificity factor", \
           units=1 , val=2600)
        
        self.AddVar( typ='Cnts', varid='Q10_tau', \
           desc="Temperatura response of specificity factor", \
           units=1, val=2.1) 
        
        self.AddVar( typ='Cnts', varid='J_max', \
           desc="Maximum electron transport rate", \
           units= mu_mol_phot * (m**-2) * d**-1, val=400)
        
        self.AddVar( typ='Cnts', varid='ab', \
           desc="Leafs absorbance", \
           units=1 , val=0.85)
        
        self.AddVar( typ='Cnts', varid='f', \
           desc="Correction factor for the spectral quality of the light", \
           units=1 , val=0.15)
        
        self.AddVar( typ='Cnts', varid='theta', \
           desc="Empirical factor", \
           units=1 , val=theta[0])

    def RHS(self, Dt):
        """RHS( Dt ) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
           
           Use from ModMod TranslateArgNames() for guide you how call the functions 
        """
        ## Cálculos de la resitencia estomática
        f_R1 = f_R( I=self.V('PAR'), C_ev1=self.V('C_ev1'), C_ev2=self.V('C_ev2') )
        Sr1 = Sr( I=self.V('PAR'), S=self.V('S'), Rs=self.V('Rs') )
        C_ev31 = C_ev3( C_ev3n=self.V('C_ev3n'), C_ev3d=self.V('C_ev3d'), Sr=Sr1 )
        f_C1 = f_C( C_ev3=C_ev31, C1=self.V('C1'), k_fc=self.V('k_fc') ) 
        C_ev41 = C_ev4( C_ev4n=self.V('C_ev4n'), C_ev4d=self.V('C_ev4d'), Sr=Sr1 )
        V_sa1 = V_sa( T =self.V('T') ) # V1 -> nos interesa
        VPD1 = VPD( V_sa=V_sa1, RH=self.V('RH') )
        f_V1 = f_V( C_ev4=C_ev41, VPD = VPD1)
        R_s1 = r_s( r_m=self.V('r_m'), f_R=f_R1, f_C=f_C1, f_V=f_V1, k_d=self.V('k_d') ) 
        ## Cálculos absorción de CO2
        g_s = gTC( k=self.V('ks'), Rb=self.V('Rb'), Rs=R_s1, k_d=self.V('k_d') )
        Ca1 = Ca( gtc=g_s, C=self.V('C1'), Ci=self.Vk('Ci') )
        Dt_Ci = ( Ca1 - (1e-3)*self.V('A') )/0.554 # Los asimilados se pasan a mg/m**2 y el incremento del Ci queda en ppm
        return Dt_Ci

    #################################################################
    ############ Módulo de crecimiento para una planta ##############
    #################################################################
class Plant(Module):
    def __init__( self, beta, Q_rhs_ins, Ci_rhs_ins, Dt_f=0.1, Dt_g=1):  # Dt_f=0.1, Dt_g=0.25
        """Models one plant growth, with a variable number of fruits."""
        ## Dt_f is the photosynthesis Dt, this is use for RK advance
        ## Dt_g is the growth Dt, this is use for the update and anvance of the fruits
        super().__init__(Dt_f) #Time steping of module, days
        ### Always, use the super class __init__, there are several other initializations
        
        self.Dt_g = Dt_g
        self.beta = beta ## must be in (0,1]
    
        ### Time units= hours
        
        ### Vegetative part
        self.veget = [0.0 , 0.0] ## characteristics for vegetative part: Weight and growth potential 

        self.fruits = [] # No fruits
        self.n_fruits = 0 ## Current number of fruits
        self.n_fruits_h = 0 ## total number of fruits harvested
        self.new_fruit = 0  ## Cummulative number of fruits
        self.m = 4 ## Number of characteristics for each fruit: thermic age, weight, growth potential and Michaelis-Menten constant
        ### Module specific constructors, add RHS's

        self.AddStateRHS( 'Q', Q_rhs_ins)
        self.AddStateRHS( 'Ci', Ci_rhs_ins)

    def Advance( self, t1):
        """Update the plant/fruit growth. Update global variables, to time t1."""
        
        ### This creates a set of times knots, that terminates in t1 with
        ### a Deltat <= self.Dt
        tt = append( arange( self.t(), t1, step=self.Dt_g), [t1])
        #print(tt)
        steps = len(tt)
        for i in range( 1, steps):
            
            # Reset the assimilation rate
            rs = 0.0
            self.V_Set('A', rs)
            
            ## Assimilation rate update
            V_cmax1 = V_cmax( T_f=self.V('T'), V_cmax25=self.V('V_cmax25'), Q10_Vcmax=self.V('Q10_Vcmax'), k_T=self.V('k_T'), k_d=self.V('k_d') )
            R_d1 = R_d( V_cmax=V_cmax1 )
            tau1 = tau( T_f=self.V('T'), tau_25=self.V('tau_25'), Q10_tau=self.V('Q10_tau'), k_T=self.V('k_T') )
            K_C1 = K_C( T_f=self.V('T'), K_C25=self.V('K_C25'), Q10_KC=self.V('Q10_KC'), k_T=self.V('k_T') )
            K_O1 = K_O( T_f=self.V('T'), K_O25=self.V('K_O25'), Q10_KO=self.V('Q10_KO'), k_T=self.V('k_T') )
            Gamma_st1 = Gamma_st( T_f=self.V('T') )
            I_21 = I_2( I =self.V('PAR'), f=self.V('f'), ab=self.V('ab') )
            J1 = J( I_2=I_21, J_max=self.V('J_max'), theta=self.V('theta'), k_d=self.V('k_d') )
            A_R1 = A_R( O_a=self.V('O_a'), tau=tau1, C_i=self.V('Ci'), V_cmax=V_cmax1, Gamma_st=Gamma_st1, K_C=K_C1, K_O=K_O1, phi=self.V('phi') )
            A_f1 = A_f( C_i=self.V('Ci'), Gamma_st=Gamma_st1, J=J1, k_JV=self.V('k_JV') )
            A_acum1 = A_acum( V_cmax=V_cmax1 )
            A1 = abs( A( A_R=A_R1, A_f=A_f1, A_acum=A_acum1, R_d=R_d1, fc=self.V('fc') ) )
            self.V_Set('A', A1)
            
            ### Check if a fruit has reched is maximum thermic age, then harvest it
            harvest = []
            wk = 0.0
            nfk = 0
            for h, fruit in enumerate(self.fruits): # h is the indice and fruit is the object
                if (fruit[0] > 275 or fruit[1]>360): # It is harvested when a fruit reaches a thermic age of 275 °C d or if the fruit's weigth is greater than 360 g
                    harvest += [h]
                    self.n_fruits -= 1 # number fruits in crop
                    self.n_fruits_h += 1 # accumulated number fruits harvested
                    w = self.V( 'Q_h') + fruit[1] # accumulated weight fruits harvested
                    self.V_Set( 'Q_h', w)
                    wk += fruit[1] # weigth fruits harvested in this moment
                    nfk += 1 # number fruits harvested in this moment
            [self.fruits.pop(i) for i in harvest]# Harvested fruits are removed from the list
            self.V_Set( 'n_k', nfk)
            self.V_Set( 'h_k', wk)
            
            ### With the Floration Rate, create new fruits
            PA_mean_i = self.beta * self.V('PAR')
            self.new_fruit += TF( k1_TF=self.V('k1_TF'), k2_TF=self.V('k2_TF'), k3_TF=self.V('k3_TF'),\
                    PA_mean=PA_mean_i, T_mean=self.V('T'), Dt=tt[i]-tt[i-1])
            new_fruit_n = self.new_fruit 
            if new_fruit_n >= 1:
                #nw = new_fruit_n
                nw = int(floor(self.new_fruit))
                for nf in range(nw):
                    ### Add new fruit
                    self.fruits += [[ 0.0, 0.0, 0.0, 0.0]] 
                    ### also the growth potential, as an auxiliar for calculations
                    self.n_fruits += 1 
                ### Leave the rational part of new_fruit
                self.new_fruit = max( 0, self.new_fruit - nw)
            
            ### Update thermic age of all fruits
            for fruit in self.fruits:
                fruit[0] += ( max( 0 , self.V('T') - 10 ) )* (tt[i]-tt[i-1]) ## Thermic age never decreases
            
            ### Update growth potencial for vegetative part
            self.veget[1] = self.V('a') + self.V('b')*self.V('T') 
            ### Update Growth potential and Michaelis-Menten constants of all fruits
            tmp = 0.0
            tmp1 = self.veget[1] / self.V('A') # start with the growth potencial of vegetative part
            for fruit in self.fruits:
                x = fruit[0] ## Thermic age
                ### Michaelis-Menten constants 
                if x <= self.V('C_t') :
                    fruit[3] = 0.05*tmp*(self.V('C_t') - x) / self.V('C_t')
                ### Growth potential
                fruit[2] = clip( Y_pot( k2_TF=self.V('k2_TF'), C_t=self.V('C_t'),\
                     B=self.V('B'), D=self.V('D'), M=self.V('M'), X=x, T_mean=self.V('T')),\
                     a_min=0, a_max=exp(300))
                tmp += fruit[2]
                tmp1 += fruit[2] / ( fruit[3] + self.V('A') )
            #self.V_Set( 'Y_sum', tmp)
            
            ### Update weight of vegetative part
            f_wg_veg =  self.veget[1] / ( self.V('A') * tmp1 ) # The sink strentgh of vegetative part
            self.veget[0] += t_wg( dw_ef=self.V('dw_ef_veg'), A=self.V('A'), f_wg=f_wg_veg) * (tt[i]-tt[i-1])
            #### Update weight of all fruits
            tmp2 = 0.0
            Dt = (tt[i]-tt[i-1])
            for fruit in self.fruits:
                f_wg =  fruit[2] / ( ( fruit[3] + self.V('A') ) * tmp1 ) # The sink strentgh of the fruit
                dwh = t_wg( dw_ef=self.V('dw_ef'), A=self.V('A'), f_wg=f_wg) * Dt # dry weight
                pdw = 0.023 # percentage of dry weight
                fruit[1] += dwh / pdw # Fresh weight  
                tmp2 += fruit[1] #Total weight
            
            #### Update assimilation rate after distribution
            m = ( f_wg_veg / self.V('dw_ef_veg') ) + ( (1 - f_wg_veg ) / self.V('dw_ef') )
            As = self.V('A')*( 1 - self.V('a_ef')*Dt*m ) # A = A - ( Total weigth of fruits and vegetative part )
            self.V_Set('A', As )  
            
            #### Total weight of the fruits
            self.V_Set( 'Q', tmp2)
            
            #### Advance of the RHS
            self.AdvanceAssigment(t1) # Set Q
            self.AdvanceRungeKutta(t1) # Set A
            
        return 1
    
    #################################################################
    ############ Director esclavo de una planta #####################
    #################################################################
def PlantDirector( beta, return_Q_rhs_ins=False):
    """Build a Director to hold a Plant, with beta PAR parameter."""


    ### Start model with empty variables
    Dir = Director( t0=0.0, time_unit="", Vars={}, Modules={} )
    
    ### Add the photosynthesis module:
    Ci_rhs_ins = Ci_rhs()
    
    Dir.AddTimeUnit( Ci_rhs_ins.GetTimeUnits())

    Dir.MergeVarsFromRHSs( Ci_rhs_ins, call=__name__)
    
    ### Start new plant  rhs
    Q_rhs_ins = Q_rhs()

    Dir.AddTimeUnit( Q_rhs_ins.GetTimeUnits())

    Dir.MergeVarsFromRHSs( Q_rhs_ins, call=__name__)
    
    ### Add an instance of Module 1:
    Dir.AddModule( "Plant", Plant( beta, Q_rhs_ins, Ci_rhs_ins) )
    Dir.sch = [ "Plant" ]

    if return_Q_rhs_ins:
        return Dir, Q_rhs_ins
    else:
        return Dir
    ################################################################
    ########### Director principal del invernadero #################
    ################################################################

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
        for _, plant in enumerate(Model.PlantList):
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


    def update_state(self, C1, T, PAR):
        self.Vars['C1'].val = C1 
        self.Vars['T'].val = T
        self.Vars['PAR'].val = PAR
        



#################################################################
############ Módulo principal del invernadero #################
#################################################################
Model = GreenHouse()

## Read data 
#U = np.ones(10)
#DirClimate = 
#Model.AddDirectorAsModule( "Climate", DirClimate)
#Model.AddModule( 'Climate', ReadModule( "Read_Data.xls", t_conv_shift=0.0, t_conv=1/(60*24)  ) )   # t_conv=1/(60*24) es para pasar el tiempo de minutos (como queda después de la lectura de la base) a días 
#Model.sch += ['Climate']
## Add global variables
### Run:
#Model.Run( Dt=1, n=nmrec, sch= Model.sch)

#return Model.OutVar('NF'), Model.OutVar('H')
# return Model
'''
    Dir = Director( t0=0.0, time_unit=symb_time_units, Vars={}, Modules={} )
        Dir.MergeVarsFromRHSs( [ C1_rhs_ins, V1_rhs_ins, T1_rhs_ins, T2_rhs_ins], call=__name__)
        Dir.AddModule( 'Climate', ReadModule( base_inputs, t_conv_shift=0.0, t_conv=1. ) ) 
        Dir.AddModule( 'Module1', Module1() )
        Dir.sch = [ 'Climate', 'Module1' ]
    '''
    ##################################################################
    ######### Lectura datos de producción Holanda ####################
    ##################################################################

'''
from matplotlib.pyplot import hist, figure, xlabel, ylabel, axvline, plot, title,show

#theta_nm = array([0.7, 3.3, 0.25]) # theta nominal
Res = Model.OutVar('NF'), Model.OutVar('H')
NF = Res[0] 
H = Res[1]
## Correcciones
# A los valores iniciales simulados
H[22:33] = H[9:20]
NF[22:33] = NF[9:20]
H[0:22] = 0
NF[0:22] = 0 
## Gráficos
t = arange(1,115+1)
plot(t, H)
xlabel('Day')
ylabel(r'$g$')
title(r'$H$')
figure()
plot(t, NF)
xlabel('Day')
ylabel('Number fruits')
title(r'$NF$')
show()
'''