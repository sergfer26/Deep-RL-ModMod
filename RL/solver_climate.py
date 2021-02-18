#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:43:58 2020

@author: jdmolinam
"""


##################################################################
##################################################################
################# Solver - Modelo Climático ##################
##################################################################
##################################################################

###########################
####### Módulos  ##########
###########################
import numpy as np
from ModMod import StateRHS, Module, Director, ReadModule
from sympy import symbols
from math import sqrt, exp, log, pi
from scipy.stats import norm
from numpy.random import seed

###########################
### Símbolos y funciones ##
###########################
######### C1 ##############
# Símbolos
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, g, mol_CH2O = symbols('mt mg m C s W mg_CO2 J g mol_CH2O')  

# previous functions


def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*(U1*f5 + (1-U1)*f5) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*(U1*f7 + (1-U1)*f7) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt(max((omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0))


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)

### functions of this RHS ###


def kappa4(phi2):
    return phi2


def o1(eta13, h6):
    return eta13*h6


def o2(U10, psi2, alpha6):
    return U10 * psi2 / alpha6


def o3(C1, I10, f1):
    return f1 * (C1 - I10)


def o4(I11, I12, I13, psi3):
    return psi3*I11*(I12 - I13)


def o5(C1, I10, f2, f3, f4):
    return (C1 - I10) * (f2 + f3 + f4)


######### V1 ##############
# Symbols
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, kmol, kg_air, kg_vapour = symbols(
    'mt mg m C s W mg_CO2 J Pa kg_water kg K ppm kmol kg_air kg_vapour')  # Symbolic use of base phisical units

### previous functions ###


def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*(U1*f5 + (1-U1)*f5) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*(U1*f7 + (1-U1)*f7) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt(max((omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0))


def h3(T2, V1, U3, I6, lamb1, lamb2, alpha6, gamma2, q6):
    num = (U3*lamb1*lamb2/alpha6)*(I6-T2)
    den = T2 - I6 + (6.4e-9 * gamma2*(V1 - q6))
    return num/den


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)


def q1(I1, rho3, alpha5, gamma, gamma2, gamma3, q3):
    return (2*rho3*alpha5*I1) / (gamma*gamma2*(gamma3 + q3))


def q2(T1):
    try:
        if (T1 > 0):
            return 0.61121*exp((18.678 - (T1/234.5)) * (T1/(257.14+T1)))
        else:
            return 0.61115*exp((23.036 - (T1/333.7)) * (T1/(279.82+T1)))
    except:
        return 14   # Esta opción es por si ocurre un error numérico. 14 es el valor inicial de V1


def q3(I14, gamma4, q4, q5, q10):
    return gamma4*q10*q4*q5


def q4(C1, eta4, q8):
    return 1 + q8*((eta4*C1 - 200)**2)


def q5(V1, q2, q9):
    return 1 + q9*((q2 - V1)**2)


def q6(I6):
    if (I6 > 0):
        return 0.61121*exp((18.678 - (I6/234.5)) * (I6/(257.14+I6)))
    else:
        return 0.61115*exp((23.036 - (I6/333.7)) * (I6/(279.82+I6)))


def q7(I14, delta1, gamma5):
    return (1 + exp(gamma5*(I14 - delta1)))**-1


def q8(delta4, delta5, q7):
    return delta4*(1 - q7) + delta5*q7


def q9(delta6, delta7, q7):
    return delta6*(1 - q7) + delta7*q7


def q10(I14, delta2, delta3):
    return (I14 + delta2)/(I14 + delta3)

### functions of this RHS ###


def kappa3(T2, psi1, phi2, omega2):
    return (psi1*phi2) / (omega2*(T2+273.15))


def p1(V1, q1, q2):
    return q1*(q2 - V1)


def p2(rho3, eta5, phi5, phi6, f1):
    return rho3*f1*(eta5*(phi5 - phi6) + phi6)


def p3(U9, phi9, alpha6):
    return U9*phi9/alpha6


def p4(eta12, h6):
    return eta12*h6


def p5(T2, V1, I5, psi1, omega2, f2, f3, f4):
    return (psi1/omega2)*((V1/(T2+273.15)) - (I5/(I5+273.15)))*(f2 + f3 + f4)


def p6(T2, V1, psi1, omega2, f1):
    return f1*(psi1/omega2)*(V1/(T2+273.15))


def p7(V1, h3, q6):
    if (V1 < q6):
        return 0
    else:
        return (6.4e-9)*h3*(V1 - q6)


######### T1 ##############
# Symbols
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm = symbols(
    'mt mg m C s W mg_CO2 J Pa kg_water kg K ppm')  # Symbolic use of base phisical units

### previous functions ###


def a1(I1, beta3):
    return 1 - exp(-beta3*I1)


def b1(U1, tau3):
    return 1 - U1*(1 - tau3)


def r2(I1, beta1, rho1, r4):
    return r4*(1 - rho1)*(1 - exp(-beta1*I1))


def r3(I1, beta1, beta2, rho1, rho2, r4):
    return r4*exp(-beta1*I1)*rho2*(1 - rho1)*(1 - exp(-beta2*I1))


def r4(I2, eta1, eta2, tau1):
    return (1 - eta1)*tau1*eta2*I2


def g1(a1):
    return 0.49*a1


def g2(tau2, b1):
    return tau2*b1


def p1(V1, q1, q2):
    return q1*(q2 - V1)


def q1(I1, rho3, alpha5, gamma, gamma2, gamma3, q3):
    return (2*rho3*alpha5*I1) / (gamma*gamma2*(gamma3 + q3))


def q2(T1):
    try:
        if (T1 > 0):
            return 0.61121*exp((18.678 - (T1/234.5)) * (T1/(257.14+T1)))
        else:
            return 0.61115*exp((23.036 - (T1/333.7)) * (T1/(279.82+T1)))
    except:
        return 14


def q3(I14, gamma4, q4, q5, q10):
    return gamma4*q10*q4*q5


def q4(C1, eta4, q8):
    return 1 + q8*((eta4*C1 - 200)**2)


def q5(V1, q2, q9):
    return 1 + q9*((q2 - V1)**2)


def q7(I14, delta1, gamma5):
    return (1 + exp(gamma5*(I14 - delta1)))**-1


def q8(delta4, delta5, q7):
    return delta4*(1 - q7) + delta5*q7


def q9(delta6, delta7, q7):
    return delta6*(1 - q7) + delta7*q7


def q10(I14, delta2, delta3):
    return (I14 + delta2)/(I14 + delta3)

### functions of this RHS ###


def kappa1(I1, alpha1):
    return alpha1*I1


def r1(r2, r3):
    return r2 + r3


def r5(I2, alpha2, eta1, eta3):
    return (1 - eta1)*alpha2*eta3*I2


def r6(T1, I3, alpha3, epsil1, epsil2, lamb, g1):
    return alpha3*epsil1*epsil2*g1*lamb*((I3 + 273.15)**4 - (T1 + 273.15)**4)


def h1(T1, T2, I1, alpha4):
    return 2*alpha4*I1*(T1-T2)


def l1(gamma2, p1):
    return gamma2*p1


def r7(T1, I4, epsil2, epsil3, lamb, a1, g2):
    return a1*epsil2*epsil3*g2*lamb*((T1 + 273.15)**4 - (I4 + 273.15)**4)


######### T2 ##############
#### Symbols ########
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, m_cover, kg_air = symbols(
    'mt mg m C s W mg_CO2 J Pa kg_water kg K ppm m_cover kg_air')  # Symbolic use of base phisical units

### previous functions ###


def b1(U1, tau3):
    return 1 - U1*(1 - tau3)


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*(U1*f5 + (1-U1)*f5) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*(U1*f7 + (1-U1)*f7) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt(max((omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0))


def g3(I1, beta3, gamma1, phi1, tau1, b1):
    return tau1*b1*(1 - 0.49*pi*gamma1*phi1)*exp(-beta3*I1)


def g4(U1, tau2):
    return U1*tau2


def h8(T2, I5, I8, alpha6, lamb5, lamb6, lamb7, lamb8):
    return lamb5*(lamb6 + lamb7*(I8**lamb8))*(T2 - I5)/alpha6


def h9(T2, I5, alpha5, rho3, f4):
    return rho3*alpha5*f4*(T2 - I5)


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)


def q6(I6):
    if (I6 > 0):
        return 0.61121*exp((18.678 - (I6/234.5)) * (I6/(257.14+I6)))
    else:
        return 0.61115*exp((23.036 - (I6/333.7)) * (I6/(279.82+I6)))


def r9(I2, alpha8, alpha9, eta2, eta3):
    return (alpha8*eta2 + alpha9*eta3)*I2


def r11(T2, I4, lamb, epsil3, epsil4, g3):
    return epsil4*epsil3*g3*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)


def r12(T2, I4, lamb, epsil3, epsil5, g4):
    return epsil5*epsil3*g4*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)


def r13(T2, I4, lamb, epsil3, epsil6):
    return epsil6*epsil3*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)

### functions of this RHS ###


def kappa2(alpha5, rho3, phi2):
    return phi2*rho3*alpha5


def h1(T1, T2, I1, alpha4):
    return 2*alpha4*I1*(T1-T2)


def h2(I5, alpha5, gamma2, eta5, rho3, phi5, phi6, f1):
    return f1*(rho3*alpha5*I5 - gamma2*rho3*(eta5*(phi5 - phi6)))


def h3(T2, V1, U3, I6, lamb1, lamb2, alpha6, gamma2, q6):
    num = (U3*lamb1*lamb2/alpha6)*(I6-T2)
    den = T2 - I6 + (6.4e-9 * gamma2*(V1 - q6))
    return num/den


def h4(T2, I3, gamma1, phi1):
    return (1.99*pi*phi1*gamma1*(abs(I3 - T2)**0.32))*(I3 - T2)


def h5(T2, I7, lamb3):
    return lamb3*(I7 - T2)


def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def r8(I2, alpha2, alpha7, eta1, eta2, eta3, tau1, r9):
    return eta1*I2*(tau1*eta2 + (alpha2 + alpha7)*eta3) + r9


def h7(T2, I5, alpha5, rho3, f2, f3, h8, h9):
    return rho3*alpha5*(f2 + f3)*(T2 - I5) + h8 + h9


def h10(T2, alpha5, rho3, f1):
    return f1*rho3*alpha5*T2


def l2(U9, alpha6, gamma2, phi9):
    return gamma2*U9*phi9/alpha6


def r10(r11, r12, r13):
    return r11 + r12 + r13


def h11(T2, I7, nu7, nu8, phi2):
    return 2*nu7*(T2 - I7)/(phi2 + nu8)


C1_in = 432
V1_in = 14
T2_in = 20
T1_in = 20
nmrec = 1

I1 = 3.5 # Constant
I2 = 0.0
I3 = 20 # Constant
I4 = 1 # Constant
I5 = 0.0
I6 = 16 # Constant
I7 = 10 # Constant
I8 = 0.0
# I9
I10 = 400 # Constant
I11 = 1 # Constant?
I12 = 1 # Constant?
I13 = 0 # Constant?
I14 = 0.0

S = [C1_in, V1_in, T1_in, T2_in]
U = np.ones(10)
theta = np.array([2000, 15, 1.9e5])

"""
    Se resuelve el sistema de EDO en función de los parámetros
    contenidos en el vector theta.
    theta = [alpha1, phi2, psi2].
    nmrec es el número de resultados de cada variable que serán reportados.
    La función retorna resultados en este orden: (C1, V1, T1, T2).
    """
u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = U
C1_in, V1_in, T1_in, T2_in = S

   ###########################
   ##### Definición RHS ######
   ###########################
   ########## C1 ############
class C1_rhs(StateRHS):
    """Define a RHS, this is the rhs for C1, the CO2 concentrartion in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec)  # falta valor inicial
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec)  # falta valor inicial
        # control variables  ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U10', prn=r'$U_{10}$',
                    desc="Control of external CO2 source", units=1, val=u10)
        self.AddVar(typ='Cnts', varid='U4', prn=r'$U_4$',
                    desc="Air heater control", units=1, val=u4)
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=u2)
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=u1)
        self.AddVar(typ='Cnts', varid='U7', prn=r'$U_7$',
                    desc="Forced ventilation control", units=1, val=u7)
        self.AddVar(typ='Cnts', varid='U8', prn=r'$U_8$',
                    desc="Roof vents control", units=1, val=u8)
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=u5)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=u6)
        # Inputs ---> No son constantes sino variables
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='Cnts', varid='I10', prn=r'$I_{10}$',
                    desc="Outdoor CO2 concentration", units=mg * m**-3, val=I10)
        self.AddVar(typ='Cnts', varid='I11', prn=r'$I_{11}$',
                    desc="Inhibition of the rate of photosynthesis by saturation of the leaves with carbohydrates", units=1, val=I11)  # Falta valor y unidades
        self.AddVar(typ='Cnts', varid='I12', prn=r'$I_{12}$',
                    desc="Crude canopy photosynthesis rate", units=1, val=I12)  # Falta valor y unidades
        self.AddVar(typ='Cnts', varid='I13', prn=r'$I_{13}$',
                    desc="Photorespiration during photosynthesis", units=1, val=I13)  # Falta valor y unidades
        # Constants
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=1)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=1e4)  # ok
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=16.7)  # ok
        self.AddVar(typ='Cnts', varid='eta6', prn=r'$\eta_6$',
                    desc="Ventilation power reduction factor", units=m**3 * m**-2 * s**-1, val=1)  # Falta valor
        self.AddVar(typ='Cnts', varid='eta7', prn=r'$\eta_7$',
                    desc="Ratio between ceiling ventilation area and total ventilation area", units=1, val=0.5)  # no dan valor en el artículo
        self.AddVar(typ='Cnts', varid='eta8', prn=r'$\eta_8$',
                    desc="Ratio between ceiling and total ventilation area, if there is no chimney effect", units=1, val=0.9)  # ok
        self.AddVar(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=0)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=1e-4)  # ok
        self.AddVar(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=2e3)  # 0.2*alpha6 --> ok
        self.AddVar(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=9.81)  # ok
        self.AddVar(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=1)  # ok
        self.AddVar(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=0.65)  # ok
        self.AddVar(typ='Cnts', varid='eta10', prn=r'$\eta_{10}$',
                    desc="Shadow effect on the discharge coefficient", units=1, val=0)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=0)  # ok, en ejemplos del artículo usan valor cero
        self.AddVar(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=0.1)  # ok
        self.AddVar(typ='Cnts', varid='eta11', prn=r'$\eta_{11}$',
                    desc="Effect of shadow on the global wind pressure coefficient", units=1, val=0)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1])  # Valor original 4
        self.AddVar(typ='Cnts', varid='eta13', prn=r'$\eta_{13}$',
                    desc="Amount of CO2 that is released when a joule of sensible energy is produced by the direct air heater", units=mg_CO2 * J**-1, val=0.057)  # ok
        self.AddVar(typ='Cnts', varid='psi2', prn=r'$\psi_2$',
                    desc="Capacity of the external CO2 source", units=mg * s**-1, val=theta[2])  # Falta valor, tomé el del ejemplo de Texas 4.3e5
        self.AddVar(typ='Cnts', varid='psi3', prn=r'$\psi_3$',
                    desc="Molar mass of the CH2O", units=g * mol_CH2O**-1, val=30.031)  # ok
    def RHS(self, Dt):
        """RHS( Dt, k) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
        """
        # Direct usage, NB: State variables need to used Vk, so that X+k is evaluated.
        # This can be done with TranslateArgNames(h1)
        # Once defined h1 in your terminal run TranslateArgNames(h1)
        # and follow the instrucions
        #### Sub-functions ####
        h_6 = h6(U4=self.V('U4'), lamb4=self.V(
            'lamb4'), alpha6=self.V('alpha6'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        f_3 = f3(U7=self.V('U7'), phi8=self.V(
            'phi8'), alpha6=self.V('alpha6'))
        f_6 = f6(I8=self.V('I8'), nu4=self.V('nu4'))
        n_1 = n1(U5=self.V('U5'), nu1=self.V('nu1'), eta10=self.V('eta10'))
        n_2 = n2(U6=self.V('U6'), nu3=self.V('nu3'))
        n_3 = n3(U5=self.V('U5'), nu2=self.V('nu2'), eta11=self.V('eta11'))
        f_5 = f5(I8=self.V('I8'), alpha6=self.V(
            'alpha6'), n1=n_1, n2=n_2, n3=n_3)
        f_2 = f2(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f5=f_5, f6=f_6)
        f_7 = f7(T2=self.Vk('T2'), U8=self.V('U8'), I5=self.V('I5'), I8=self.V('I8'), nu5=self.V(
            'nu5'), alpha6=self.V('alpha6'), omega1=self.V('omega1'), nu6=self.V('nu6'), n1=n_1, n3=n_3)
        f_4 = f4(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f6=f_6, f7=f_7)
        #### Principal functions ####
        kappa_4 = kappa4(phi2=self.V('phi2'))
        o_1 = o1(eta13=self.V('eta13'), h6=h_6)
        o_2 = o2(U10=self.V('U10'), psi2=self.V(
            'psi2'), alpha6=self.V('alpha6'))
        o_3 = o3(C1=self.Vk('C1'), I10=self.V('I10'), f1=f_1)
        o_4 = o4(I11=self.V('I11'), I12=self.V('I12'),
                 I13=self.V('I13'), psi3=self.V('psi3'))
        o_5 = o5(C1=self.Vk('C1'), I10=self.V(
            'I10'), f2=f_2, f3=f_3, f4=f_4)
        return (kappa_4**-1)*(o_1 + o_2 + o_3 - o_4 - o_5)


########### V1 ############
class V1_rhs(StateRHS):
    """Define a RHS, this is the rhs for V1, the vapour pression in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec) # falta valor inicial
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U4', prn=r'$U_4$',
                    desc="Air heater control", units=1, val=u4)
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=u2)
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=u1)
        self.AddVar(typ='Cnts', varid='U7', prn=r'$U_7$',
                    desc="Forced ventilation control", units=1, val=u7)
        self.AddVar(typ='Cnts', varid='U8', prn=r'$U_8$',
                    desc="Roof vents control", units=1, val=u8)
        self.AddVar(typ='Cnts', varid='U3', prn=r'$U_3$',
                    desc="Control of mechanical cooling system", units=1, val=u3)
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=u5)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=u6)
        self.AddVar(typ='Cnts', varid='U9', prn=r'$U_9$',
                    desc="Fog system control", units=1, val=u9)
        # Inputs
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val=I6)
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=I14)
        # Constants
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=1) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=1e4) # ok
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=16.7) # ok
        self.AddVar(typ='Cnts', varid='eta6', prn=r'$\eta_6$',
                    desc="Ventilation power reduction factor", units=m**3 * m**-2 * s**-1, val=1) # Falta valor
        self.AddVar(typ='Cnts', varid='eta7', prn=r'$\eta_7$',
                    desc="Ratio between ceiling ventilation area and total ventilation area", units=1, val=0.5) # no dan valor en el artículo
        self.AddVar(typ='Cnts', varid='eta8', prn=r'$\eta_8$',
                    desc="Ratio between ceiling and total ventilation area, if there is no chimney effect", units=1, val=0.9) # ok
        self.AddVar(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=1e-4) # ok
        self.AddVar(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=2e3) # 0.2*alpha6 --> ok
        self.AddVar(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=9.81) # ok
        self.AddVar(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=1) # ok
        self.AddVar(typ='Cnts', varid='lamb1', prn=r'$\lambda_1$',
                    desc="Performance coefficient of the mechanical acceleration system", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb2', prn=r'$\lambda_2$',
                    desc="Electrical capacity of the mechanical cooling system", units=W, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=2.45e6) # ok
        self.AddVar(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=0.65) # ok
        self.AddVar(typ='Cnts', varid='eta10', prn=r'$\eta_{10}$',
                    desc="Shadow effect on the discharge coefficient", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=0) # ok, en ejemplos del artículo usan valor cero
        self.AddVar(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=0.1) # ok
        self.AddVar(typ='Cnts', varid='eta11', prn=r'$\eta_{11}$',
                    desc="Effect of shadow on the global wind pressure coefficient", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= 1.2) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=1e3) # ok
        self.AddVar(typ='Cnts', varid='gamma', prn=r'$\gamma$',
                    desc="Psychometric constan", units=Pa * K**-1, val=65.8) # ok
        self.AddVar(typ='Cnts', varid='gamma3', prn=r'$\gamma_3$',
                    desc="Strength of boundary layer of canopy for vapor transport", units=s * m**-1, val=275) # ok
        self.AddVar(typ='Cnts', varid='delta1', prn=r'$\delta_2$',
                    desc="Radiation above the canopy that defines sunrise and sunset", units=W * m**-2, val=5) # ok
        self.AddVar(typ='Cnts', varid='delta2', prn=r'$\delta_3$',
                    desc="Empirically determined parameter", units=W * m**-2, val=4.3) # ok
        self.AddVar(typ='Cnts', varid='delta3', prn=r'$\delta_4$',
                    desc="Empirically determined parameter", units=W * m**-2, val=0.54) # ok
        self.AddVar(typ='Cnts', varid='gamma4', prn=r'$\gamma_4$',
                    desc="Minimum stomatal resistance of the canopy", units=s * m**-1, val=82.0) # ok
        self.AddVar(typ='Cnts', varid='gamma5', prn=r'$\gamma_5$',
                    desc="Slope of the differentiable switch for the stomatal resistance model", units=m * W**-2, val=-1) # ok
        self.AddVar(typ='Cnts', varid='delta4', prn=r'$\delta_5$',
                    desc="Coefficient of the CO2 transpiration in the day", units=ppm**-2, val=6.1e-7) # ok
        self.AddVar(typ='Cnts', varid='delta5', prn=r'$\delta_6$',
                    desc="Coefficient of the CO2 transpiration in the night", units=ppm**-2, val=1.1e-11) # ok
        self.AddVar(typ='Cnts', varid='delta6', prn=r'$\delta_7$',
                    desc="Coefficient of the vapour pressure in the day", units=Pa**-2, val=4.3e-6) # ok
        self.AddVar(typ='Cnts', varid='delta7', prn=r'$\delta_8$',
                    desc="Coefficient of the vapour pressure in the night", units=Pa**-2, val=5.2e-6) # ok
        self.AddVar(typ='Cnts', varid='eta4', prn=r'$\eta_4$',
                    desc="Conversion factor for CO2 of mg*m**−3 to ppm", units=ppm * mg**-1 * m**3, val=0.554) # ok
        self.AddVar(typ='Cnts', varid='psi1', prn=r'$\psi_1$',
                    desc="Molar mass of water", units=kg * kmol**-1, val=18) # ok
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1]) # Valor original 4
        self.AddVar(typ='Cnts', varid='omega2', prn=r'$\omega_2$',
                    desc="Molar gas constant", units=J * kmol**-1 * K**-1, val= 8.314e3) # ok
        self.AddVar(typ='Cnts', varid='eta5', prn=r'$\eta_5$',
                    desc="Fan-pad system efficiency", units=1, val=0) # Falta valor
        self.AddVar(typ='Cnts', varid='phi5', prn=r'$\phi_5$',
                    desc="Water vapor contained in the fan-pad system", units=kg_water * kg_air**-1, val=0) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi6', prn=r'$\phi_6$',
                    desc="Water vapor contained in the outside air", units=kg_water * kg_air**-1, val=0) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi9', prn=r'$\phi_9$',
                    desc="Fog system capacity", units=kg_water * s**-1, val=0) # Falta valor
        self.AddVar(typ='Cnts', varid='eta12', prn=r'$\eta_{12}$',
                    desc="Amount of vapor that is released when a joule of sensible energy is produced by the direct air heater", units=kg_vapour * J**-1, val=4.43e-8) # ok
    def RHS(self, Dt):

        """RHS( Dt, k) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
        """
        # Direct usage, NB: State variables need to used Vk, so that X+k is evaluated.
        # This can be done with TranslateArgNames(h1)
        # Once defined h1 in your terminal run TranslateArgNames(h1)
        # and follow the instrucions
        #### Sub-functions ####
        h_6 = h6(U4=self.V('U4'), lamb4=self.V(
            'lamb4'), alpha6=self.V('alpha6'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        f_3 = f3(U7=self.V('U7'), phi8=self.V(
            'phi8'), alpha6=self.V('alpha6'))
        f_6 = f6(I8=self.V('I8'), nu4=self.V('nu4'))
        n_1 = n1(U5=self.V('U5'), nu1=self.V('nu1'), eta10=self.V('eta10'))
        n_2 = n2(U6=self.V('U6'), nu3=self.V('nu3'))
        n_3 = n3(U5=self.V('U5'), nu2=self.V('nu2'), eta11=self.V('eta11'))
        f_5 = f5(I8=self.V('I8'), alpha6=self.V(
            'alpha6'), n1=n_1, n2=n_2, n3=n_3)
        f_2 = f2(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f5=f_5, f6=f_6)
        f_7 = f7(T2=self.Vk('T2'), U8=self.V('U8'), I5=self.V('I5'), I8=self.V('I8'), nu5=self.V(
            'nu5'), alpha6=self.V('alpha6'), omega1=self.V('omega1'), nu6=self.V('nu6'), n1=n_1, n3=n_3)
        f_4 = f4(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f6=f_6, f7=f_7)
        q_2 = q2(T1=self.Vk('T1'))
        q_7 = q7(I14=self.V('I14'), delta1=self.V(
            'delta1'), gamma5=self.V('gamma5'))
        q_8 = q8(delta4=self.V('delta4'), delta5=self.V('delta5'), q7=q_7)
        q_9 = q9(delta6=self.V('delta6'), delta7=self.V('delta7'), q7=q_7)
        q_4 = q4(C1=self.V('C1'), eta4=self.V('eta4'), q8=q_8)
        q_5 = q5(V1=self.V('V1'), q2=q_2, q9=q_9)
        q_6 = q6(I6=self.V('I6'))
        q_10 = q10(I14=self.V('I14'), delta2=self.V(
            'delta2'), delta3=self.V('delta3'))
        q_3 = q3(I14=self.V('I14'), gamma4=self.V(
            'gamma4'), q4=q_4, q5=q_5, q10=q_10)
        q_1 = q1(I1=self.V('I1'), rho3=self.V('rho3'), alpha5=self.V('alpha5'), gamma=self.V(
            'gamma'), gamma2=self.V('gamma2'), gamma3=self.V('gamma3'), q3=q_3)
        h_3 = h3(T2=self.V('T2'), V1=self.V('V1'), U3=self.V('U3'), I6=self.V('I6'), lamb1=self.V(
            'lamb1'), lamb2=self.V('lamb2'), alpha6=self.V('alpha6'), gamma2=self.V('gamma2'), q6=q_6)
        #### Principal functions ####
        kappa_3 = kappa3(T2=self.Vk('T2'), psi1=self.V(
            'psi1'), phi2=self.V('phi2'), omega2=self.V('omega2'))
        p_1 = p1(V1=self.Vk('V1'), q1=q_1, q2=q_2)
        p_2 = p2(rho3=self.V('rho3'), eta5=self.V('eta5'),
                 phi5=self.V('phi5'), phi6=self.V('phi6'), f1=f_1)
        p_3 = p3(U9=self.V('U9'), phi9=self.V(
            'phi9'), alpha6=self.V('alpha6'))
        p_4 = p4(eta12=self.V('eta12'), h6=h_6)
        p_5 = p5(T2=self.Vk('T2'), V1=self.Vk('V1'), I5=self.V('I5'), psi1=self.V(
            'psi1'), omega2=self.V('omega2'), f2=f_2, f3=f_3, f4=f_4)
        p_6 = p6(T2=self.Vk('T2'), V1=self.Vk('V1'), psi1=self.V(
            'psi1'), omega2=self.V('omega2'), f1=f_1)
        p_7 = p7(V1=self.Vk('V1'), h3=h_3, q6=q_6)
        return (kappa_3**-1)*(p_1 + p_2 + p_3 + p_4 - p_5 - p_6 - p_7)


########### T1 ############
class T1_rhs(StateRHS):
    """Define a RHS, this is the rhs for C1, the CO2 concentrartion in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec) # falta valor inicial
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=u1)
        # Inputs
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I2', prn=r'$I_2$',
                    desc="External global radiation", units=W * m**-2, val=I2)
        self.AddVar(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=I3)
        self.AddVar(typ='State', varid='I4', prn=r'$I_4$',
                    desc="Sky temperature", units=C, val=I4)
        self.AddVar(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=I14)
        # Constants
        self.AddVar(typ='Cnts', varid='beta3', prn=r'$\beta_3$',
                    desc="Canopy extinction coefficient for NIR radiation", units=1, val=0.27) # ok
        self.AddVar(typ='Cnts', varid='tau3', prn=r'$\tau_3$',
                    desc="FIR transmission coefficient of the thermal screen", units=1, val=0.11) # ok --> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='beta1', prn=r'$\beta_1$',
                    desc="Canopy extinction coefficient for PAR radiation", units=1, val=0.7) # ok
        self.AddVar(typ='Cnts', varid='rho1', prn=r'$\rho_1$',
                    desc="PAR reflection coefficient", units=1, val=0.07) # ok
        self.AddVar(typ='Cnts', varid='beta2', prn=r'$\beta_2$',
                    desc="Extinction coefficient for PAR radiation reflected from the floor to the canopy", units=1, val=0.7) # ok
        self.AddVar(typ='Cnts', varid='rho2', prn=r'$\rho_2$',
                    desc="Floor reflection coefficient PAR", units=1, val=0.65) # ok
        self.AddVar(typ='Cnts', varid='eta1', prn=r'$\eta_1$',
                    desc="Proportion of global radiation that is absorbed by greenhouse building elements", units=1, val=0.1) # ok
        self.AddVar(typ='Cnts', varid='eta2', prn=r'$\eta_2$',
                    desc="Ratio between PAR radiation and external global radiation", units=1, val=0.5) # ok
        self.AddVar(typ='Cnts', varid='tau1', prn=r'$\tau_1$',
                    desc="PAR transmission coefficient of the Cover", units=1, val=1) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='tau2', prn=r'$\tau_2$',
                    desc="FIR transmission coefficient of the Cover", units=1, val=1) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= 1.2) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=1e3) # ok
        self.AddVar(typ='Cnts', varid='gamma', prn=r'$\gamma$',
                    desc="Psychometric constan", units=Pa * K**-1, val=65.8) # ok
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=2.45e6) # ok
        self.AddVar(typ='Cnts', varid='gamma3', prn=r'$\gamma_3$',
                    desc="Strength of boundary layer of canopy for vapor transport", units=s * m**-1, val=275) # ok
        self.AddVar(typ='Cnts', varid='delta1', prn=r'$\delta_2$',
                    desc="Radiation above the canopy that defines sunrise and sunset", units=W * m**-2, val=5) # ok
        self.AddVar(typ='Cnts', varid='delta2', prn=r'$\delta_3$',
                    desc="Empirically determined parameter", units=W * m**-2, val=4.3) # ok
        self.AddVar(typ='Cnts', varid='delta3', prn=r'$\delta_4$',
                    desc="Empirically determined parameter", units=W * m**-2, val=0.54) # ok
        self.AddVar(typ='Cnts', varid='gamma4', prn=r'$\gamma_4$',
                    desc="Minimum stomatal resistance of the canopy", units=s * m**-1, val=82.0) # ok
        self.AddVar(typ='Cnts', varid='gamma5', prn=r'$\gamma_5$',
                    desc="Slope of the differentiable switch for the stomatal resistance model", units=m * W**-2, val=-1) # ok
        self.AddVar(typ='Cnts', varid='delta4', prn=r'$\delta_5$',
                    desc="Coefficient of the CO2 transpiration in the day", units=ppm**-2, val=6.1e-7) # ok
        self.AddVar(typ='Cnts', varid='delta5', prn=r'$\delta_6$',
                    desc="Coefficient of the CO2 transpiration in the night", units=ppm**-2, val=1.1e-11) # ok
        self.AddVar(typ='Cnts', varid='delta6', prn=r'$\delta_7$',
                    desc="Coefficient of the vapour pressure in the day", units=Pa**-2, val=4.3e-6) # ok
        self.AddVar(typ='Cnts', varid='delta7', prn=r'$\delta_8$',
                    desc="Coefficient of the vapour pressure in the night", units=Pa**-2, val=5.2e-6) # ok
        self.AddVar(typ='Cnts', varid='eta4', prn=r'$\eta_4$',
                    desc="Conversion factor for CO2 of mg*m**−3 to ppm", units=ppm * mg**-1 * m**3, val=0.554) # ok
        self.AddVar(typ='Cnts', varid='alpha1', prn=r'$\alpha_1$',
                    desc="Heat capacity of one square meter of the canopy", units=J * K**-1 * m**-2, val=theta[0]) # Valor original 1.2e3
        self.AddVar(typ='Cnts', varid='alpha2', prn=r'$\alpha_2$',
                    desc="Global NIR absorption coefficient of the canopy", units=1, val=0.35) # ok
        self.AddVar(typ='Cnts', varid='eta3', prn=r'$\eta_3$',
                    desc="Ratio between NIR radiation and global external radiation", units=1, val=0.5) # ok
        self.AddVar(typ='Cnts', varid='alpha3', prn=r'$\alpha_3$',
                    desc="Surface of the heating pipe", units=m**2, val=1) # Falta valor
        self.AddVar(typ='Cnts', varid='epsil1', prn=r'$\epsilon_1$',
                    desc="FIR emission coefficient of the heating pipe", units=1, val=0.88) # ok
        self.AddVar(typ='Cnts', varid='epsil2', prn=r'$\epsilon_2$',
                    desc="Canopy FIR emission coefficient", units=1, val=1) # ok
        self.AddVar(typ='Cnts', varid='lamb', prn=r'$\lambda$',
                    desc="Boltzmann constant", units=W * m**-2 * K**-4, val=5.670e-8) # ok
        self.AddVar(typ='Cnts', varid='alpha4', prn=r'$\alpha_4$',
                    desc="Convection heat exchange coefficient of canopy leaf to greenhouse air", units=W * m**-2 * K**-1, val=5) # ok
        self.AddVar(typ='Cnts', varid='epsil3', prn=r'$\epsilon_3$',
                    desc="Sky FIR emission coefficient", units=1, val=1)  # ok
    def RHS(self, Dt):
        """RHS( Dt, k) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
        """
        # Direct usage, NB: State variables need to used Vk, so that X+k is evaluated.
        # This can be done with TranslateArgNames(h1)
        # Once defined h1 in your terminal run TranslateArgNames(h1)
        # and follow the instrucions
        #### Sub-functions ####
        a_1 = a1(I1=self.V('I1'), beta3=self.V('beta3'))
        b_1 = b1(U1=self.V('U1'), tau3=self.V('tau3'))
        r_4 = r4(I2=self.V('I2'), eta1=self.V('eta1'),
                 eta2=self.V('eta2'), tau1=self.V('tau1'))
        r_2 = r2(I1=self.V('I1'), beta1=self.V(
            'beta1'), rho1=self.V('rho1'), r4=r_4)
        r_3 = r3(I1=self.V('I1'), beta1=self.V('beta1'), beta2=self.V(
            'beta2'), rho1=self.V('rho1'), rho2=self.V('rho2'), r4=r_4)
        g_1 = g1(a1=a_1)
        g_2 = g2(tau2=self.V('tau2'), b1=b_1)
        q_2 = q2(T1=self.Vk('T1'))
        q_7 = q7(I14=self.V('I14'), delta1=self.V(
            'delta1'), gamma5=self.V('gamma5'))
        q_8 = q8(delta4=self.V('delta4'), delta5=self.V('delta5'), q7=q_7)
        q_9 = q9(delta6=self.V('delta6'), delta7=self.V('delta7'), q7=q_7)
        q_4 = q4(C1=self.V('C1'), eta4=self.V('eta4'), q8=q_8)
        q_5 = q5(V1=self.V('V1'), q2=q_2, q9=q_9)
        q_10 = q10(I14=self.V('I14'), delta2=self.V(
            'delta2'), delta3=self.V('delta3'))
        q_3 = q3(I14=self.V('I14'), gamma4=self.V(
            'gamma4'), q4=q_4, q5=q_5, q10=q_10)
        q_1 = q1(I1=self.V('I1'), rho3=self.V('rho3'), alpha5=self.V('alpha5'), gamma=self.V(
            'gamma'), gamma2=self.V('gamma2'), gamma3=self.V('gamma3'), q3=q_3)
        p_1 = p1(V1=self.Vk('V1'), q1=q_1, q2=q_2)
        #### Principal functions ####
        kappa_1 = kappa1(I1=self.V('I1'), alpha1=self.V('alpha1'))
        r_1 = r1(r2=r_2, r3=r_3)
        r_5 = r5(I2=self.V('I2'), alpha2=self.V('alpha2'),
                 eta1=self.V('eta1'), eta3=self.V('eta3'))
        r_6 = r6(T1=self.Vk('T1'), I3=self.V('I3'), alpha3=self.V('alpha3'), epsil1=self.V(
            'epsil1'), epsil2=self.V('epsil2'), lamb=self.V('lamb'), g1=g_1)
        h_1 = h1(T1=self.Vk('T1'), T2=self.Vk('T2'),
                 I1=self.V('I1'), alpha4=self.V('alpha4'))
        l_1 = l1(gamma2=self.V('gamma2'), p1=p_1)
        r_7 = r7(T1=self.Vk('T1'), I4=self.V('I4'), epsil2=self.V(
            'epsil2'), epsil3=self.V('epsil3'), lamb=self.V('lamb'), a1=a_1, g2=g_2)
        return (kappa_1**-1)*(r_1 + r_5 + r_6 - h_1 - l_1 - r_7)


########### T2 ############
class T2_rhs(StateRHS):
    """Define a RHS, this is the rhs for T2, Greenhouse air temperature"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=u1)
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=u2)
        self.AddVar(typ='Cnts', varid='U7', prn=r'$U_7$',
                    desc="Forced ventilation control", units=1, val=u7)
        self.AddVar(typ='Cnts', varid='U8', prn=r'$U_8$',
                    desc="Roof vents control", units=1, val=u8)
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=u5)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=u6)
        self.AddVar(typ='Cnts', varid='U3', prn=r'$U_3$',
                    desc="Control of mechanical cooling system", units=1, val=u3)
        self.AddVar(typ='Cnts', varid='U4', prn=r'$U_4$',
                    desc="Air heater control", units=1, val=u4)
        self.AddVar(typ='Cnts', varid='U9', prn=r'$U_9$',
                    desc="Fog system control", units=1, val=u9)
        # Inputs
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val=I6)
        self.AddVar(typ='State', varid='I2', prn=r'$I_2$',
                    desc="External global radiation", units=W * m**-2, val=I2)
        self.AddVar(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=I3)
        self.AddVar(typ='State', varid='I4', prn=r'$I_4$',
                    desc="Sky temperature", units=C, val=I4)
        self.AddVar(typ='Cnts', varid='I7', prn=r'$I_7$',
                    desc="Soil temperature", units=C, val=I7) # Valor tomado de internet
        # Constants
        self.AddVar(typ='Cnts', varid='tau3', prn=r'$\tau_3$',
                    desc="FIR transmission coefficient of the thermal screen", units=1, val=0.11) # ok --> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=16.7) # ok
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=1e4) # ok
        self.AddVar(typ='Cnts', varid='eta6', prn=r'$\eta_6$',
                    desc="Ventilation power reduction factor", units=m**3 * m**-2 * s**-1, val=1) # Falta valor
        self.AddVar(typ='Cnts', varid='eta7', prn=r'$\eta_7$',
                    desc="Ratio between ceiling ventilation area and total ventilation area", units=1, val=0.5) # no dan valor en el artículo
        self.AddVar(typ='Cnts', varid='eta8', prn=r'$\eta_8$',
                    desc="Ratio between ceiling and total ventilation area, if there is no chimney effect", units=1, val=0.9) # ok
        self.AddVar(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=1e-4) # ok
        self.AddVar(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=2e3) # 0.2*alpha6 --> ok
        self.AddVar(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=9.81) # ok
        self.AddVar(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=1) # ok
        self.AddVar(typ='Cnts', varid='beta3', prn=r'$\beta_3$',
                    desc="Canopy extinction coefficient for NIR radiation", units=1, val=0.27) # ok
        self.AddVar(typ='Cnts', varid='gamma1', prn=r'$\gamma_1$',
                    desc="Length of the heating pipe", units=m * m**-2, val=1.25) # ok ---> Usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi1', prn=r'$\phi_1$',
                    desc="External diameter of the heating pipe", units=m, val=51e-3) # ok
        self.AddVar(typ='Cnts', varid='tau1', prn=r'$\tau_1$',
                    desc="PAR transmission coefficient of the Cover", units=1, val=1) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='tau2', prn=r'$\tau_2$',
                    desc="FIR transmission coefficient of the Cover", units=1, val=1) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='lamb5', prn=r'$\lambda_5$',
                    desc="Cover surface", units=m**2, val=1.8e4) # ok --> tomé el valor de Holanda, el de Texas es muy grande (9e4)
        self.AddVar(typ='Cnts', varid='lamb6', prn=r'$\lambda_6$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", units=W * m_cover**-2 * K**-1, val=2.8) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='lamb7', prn=r'$\lambda_7$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", units=J * m**-3 * K**-1, val=1.2) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='lamb8', prn=r'$\lambda_8$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", units=1, val=1) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=1e3) # ok
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= 1.2) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=0.65) # ok
        self.AddVar(typ='Cnts', varid='eta10', prn=r'$\eta_{10}$',
                    desc="Shadow effect on the discharge coefficient", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=0) # ok, en ejemplos del artículo usan valor cero
        self.AddVar(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=0.1) # ok
        self.AddVar(typ='Cnts', varid='eta11', prn=r'$\eta_{11}$',
                    desc="Effect of shadow on the global wind pressure coefficient", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha8', prn=r'$\alpha_8$',
                    desc="PAR absorption coefficient of the cover", units=1, val=1) # En el artículo no dan el valor
        self.AddVar(typ='Cnts', varid='alpha9', prn=r'$\alpha_9$',
                    desc="NIR absorption coefficient of the cover", units=1, val=1) # En el artículo no dan el valor
        self.AddVar(typ='Cnts', varid='eta2', prn=r'$\eta_2$',
                    desc="Ratio between PAR radiation and external global radiation", units=1, val=0.5) # ok
        self.AddVar(typ='Cnts', varid='eta3', prn=r'$\eta_3$',
                    desc="Ratio between NIR radiation and global external radiation", units=1, val=0.5) # ok
        self.AddVar(typ='Cnts', varid='lamb', prn=r'$\lambda$',
                    desc="Boltzmann constant", units=W * m**-2 * K**-4, val=5.670e-8) # ok
        self.AddVar(typ='Cnts', varid='epsil3', prn=r'$\epsilon_3$',
                    desc="Sky FIR emission coefficient", units=1, val=1)  # ok
        self.AddVar(typ='Cnts', varid='epsil4', prn=r'$\epsilon_4$',
                    desc="Floor FIR emission coefficient", units=1, val=1) # ok
        self.AddVar(typ='Cnts', varid='epsil5', prn=r'$\epsilon_5$',
                    desc="Thermal screen FIR emission coefficient", units=1, val=1)
        self.AddVar(typ='Cnts', varid='epsil6', prn=r'$\epsilon_6$',
                    desc="External cover FIR emission coefficient", units=1, val=0.44) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1]) # Valor original 4
        self.AddVar(typ='Cnts', varid='alpha4', prn=r'$\alpha_4$',
                    desc="Convection heat exchange coefficient of canopy leaf to greenhouse air", units=W * m**-2 * K**-1, val=5) # ok
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=2.45e6) # ok
        self.AddVar(typ='Cnts', varid='eta5', prn=r'$\eta_5$',
                    desc="Fan-pad system efficiency", units=1, val=0) # Falta valor
        self.AddVar(typ='Cnts', varid='phi5', prn=r'$\phi_5$',
                    desc="Water vapor contained in the fan-pad system", units=kg_water * kg_air**-1, val=0) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi6', prn=r'$\phi_6$',
                    desc="Water vapor contained in the outside air", units=kg_water * kg_air**-1, val=0) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='lamb1', prn=r'$\lambda_1$',
                    desc="Performance coefficient of the mechanical acceleration system", units=1, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb2', prn=r'$\lambda_2$',
                    desc="Electrical capacity of the mechanical cooling system", units=W, val=0) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb3', prn=r'$\lambda_3$',
                    desc="Convictive heat exchange coefficient between soil and greenhouse air", units=W * m**-2 * K**-1, val=1) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=1) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha2', prn=r'$\alpha_2$',
                    desc="Global NIR absorption coefficient of the canopy", units=1, val=0.35) # ok
        self.AddVar(typ='Cnts', varid='alpha7', prn=r'$\alpha_7$',
                    desc="Global NIR absorption coefficient of the floor", units=1, val=0.5) # ok
        self.AddVar(typ='Cnts', varid='eta1', prn=r'$\eta_1$',
                    desc="Proportion of global radiation that is absorbed by greenhouse building elements", units=1, val=0.1) # ok
        self.AddVar(typ='Cnts', varid='phi9', prn=r'$\phi_9$',
                    desc="Fog system capacity", units=kg_water * s**-1, val=0) # Falta valor
        self.AddVar(typ='Cnts', varid='nu7', prn=r'$\nu_7$',
                    desc="Soil thermal conductivity", units=W * m**-1 * K**-1, val=0.85) # ok
        self.AddVar(typ='Cnts', varid='nu8', prn=r'$\nu_8$',
                    desc="Floor to ground distance", units=m, val=0.64) # ok
    def RHS(self, Dt):
        """RHS( Dt, k) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
        """
        # Direct usage, NB: State variables need to used Vk, so that X+k is evaluated.
        # This can be done with TranslateArgNames(h1)
        # Once defined h1 in your terminal run TranslateArgNames(h1)
        # and follow the instrucions
        #### Sub-functions ####
        b_1 = b1(U1=self.V('U1'), tau3=self.V('tau3'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        n_1 = n1(U5=self.V('U5'), nu1=self.V('nu1'), eta10=self.V('eta10'))
        n_2 = n2(U6=self.V('U6'), nu3=self.V('nu3'))
        n_3 = n3(U5=self.V('U5'), nu2=self.V('nu2'), eta11=self.V('eta11'))
        f_5 = f5(I8=self.V('I8'), alpha6=self.V(
            'alpha6'), n1=n_1, n2=n_2, n3=n_3)
        f_6 = f6(I8=self.V('I8'), nu4=self.V('nu4'))
        f_2 = f2(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f5=f_5, f6=f_6)
        f_3 = f3(U7=self.V('U7'), phi8=self.V(
            'phi8'), alpha6=self.V('alpha6'))
        f_7 = f7(T2=self.Vk('T2'), U8=self.V('U8'), I5=self.V('I5'), I8=self.V('I8'), nu5=self.V(
            'nu5'), alpha6=self.V('alpha6'), omega1=self.V('omega1'), nu6=self.V('nu6'), n1=n_1, n3=n_3)
        f_4 = f4(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f6=f_6, f7=f_7)
        g_3 = g3(I1=self.V('I1'), beta3=self.V('beta3'), gamma1=self.V(
            'gamma1'), phi1=self.V('phi1'), tau1=self.V('tau1'), b1=b_1)
        g_4 = g4(U1=self.V('U1'), tau2=self.V('tau2'))
        h_8 = h8(T2=self.Vk('T2'), I5=self.V('I5'), I8=self.V('I8'), alpha6=self.V('alpha6'), lamb5=self.V(
            'lamb5'), lamb6=self.V('lamb6'), lamb7=self.V('lamb7'), lamb8=self.V('lamb8'))
        h_9 = h9(T2=self.Vk('T2'), I5=self.V('I5'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f4=f_4)
        q_6 = q6(I6=self.V('I6'))
        r_9 = r9(I2=self.V('I2'), alpha8=self.V('alpha8'), alpha9=self.V(
            'alpha9'), eta2=self.V('eta2'), eta3=self.V('eta3'))
        r_11 = r11(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil4=self.V('epsil4'), g3=g_3)
        r_12 = r12(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil5=self.V('epsil5'), g4=g_4)
        r_13 = r13(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil6=self.V('epsil6'))
        #### Principal functions ####
        kappa_2 = kappa2(alpha5=self.V('alpha5'),
                         rho3=self.V('rho3'), phi2=self.V('phi2'))
        h_1 = h1(T1=self.Vk('T1'), T2=self.Vk('T2'),
                 I1=self.V('I1'), alpha4=self.V('alpha4'))
        h_2 = h2(I5=self.V('I5'), alpha5=self.V('alpha5'), gamma2=self.V('gamma2'), eta5=self.V(
            'eta5'), rho3=self.V('rho3'), phi5=self.V('phi5'), phi6=self.V('phi6'), f1=f_1)
        h_3 = h3(T2=self.Vk('T2'), V1=self.Vk('V1'), U3=self.V('U3'), I6=self.V('I6'), lamb1=self.V(
            'lamb1'), lamb2=self.V('lamb2'), alpha6=self.V('alpha6'), gamma2=self.V('gamma2'), q6=q_6)
        h_4 = h4(T2=self.Vk('T2'), I3=self.V('I3'),
                 gamma1=self.V('gamma1'), phi1=self.V('phi1'))
        h_5 = h5(T2=self.Vk('T2'), I7=self.V('I7'), lamb3=self.V('lamb3'))
        h_6 = h6(U4=self.V('U4'), lamb4=self.V(
            'lamb4'), alpha6=self.V('alpha6'))
        r_8 = r8(I2=self.V('I2'), alpha2=self.V('alpha2'), alpha7=self.V('alpha7'), eta1=self.V(
            'eta1'), eta2=self.V('eta2'), eta3=self.V('eta3'), tau1=self.V('tau1'), r9=r_9)
        h_7 = h7(T2=self.Vk('T2'), I5=self.V('I5'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f2=f_2, f3=f_3, h8=h_8, h9=h_9)
        h_10 = h10(T2=self.Vk('T2'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f1=f_1)
        l_2 = l2(U9=self.V('U9'), alpha6=self.V('alpha6'),
                 gamma2=self.V('gamma2'), phi9=self.V('phi9'))
        r_10 = r10(r11=r_11, r12=r_12, r13=r_13)
        h_11 = h11(T2=self.Vk('T2'), I7=self.V('I7'), nu7=self.V(
            'nu7'), nu8=self.V('nu8'), phi2=self.V('phi2'))
        return (kappa_2**-1)*(h_1 + h_2 + h_3 + h_4 + h_5 + h_6 + r_8 - h_7 - h_10 - l_2 - r_10 - h_11)


###########################
### Definición Director ###
###########################
class Module1(Module):

    def __init__(self, Dt=1, **kwargs):
        """Models one part of the process, uses the shared variables
           from Director.
           Dt=0.1, default Time steping of module
        """
        super().__init__(Dt)  # Time steping of module
        # Always, use the super class __init__, theare are several otjer initializations
        # Module specific constructors, add RHS's
        for key, value in kwargs.items():
            self.AddStateRHS(key, value)
        # print("State Variables for this module:", self.S_RHS_ids)

    def Advance(self, t1):
        # Se agrega ruido a los resultados del modelo
        s1 = 0.1131  # Desviación estándar de T1 y T2
        s2 = 0.1281  # Desviación estándar de V1
        s3 = 10  # Desviación estándar de C1
        # seed( int( self.t() ) ) # La semilla de los aleatorios depende del tiempo del director
        T1r = self.V('T1') + norm.rvs(scale=s1)
        T2r = self.V('T2') + norm.rvs(scale=s1)
        V1r = self.V('V1') + norm.rvs(scale=s2)
        C1r = self.V('C1') + norm.rvs(scale=s3)
        # Actualización de las variables
        self.V_Set('T1', T1r)
        self.V_Set('T2', T2r)
        self.V_Set('V1', V1r)
        self.V_Set('C1', C1r)
        # Avance del RHS
        self.AdvanceRungeKutta(t1)
        return 1
        

class Climate_model(Director):
    def __init__(self):
        super().__init__(t0=0.0, time_unit="", Vars={}, Modules={})
        C1_rhs_ins = C1_rhs()  # Make an instance of rhs
        V1_rhs_ins = V1_rhs()  # Make an instance of rhs
        T1_rhs_ins = T1_rhs()  # Make an instance of rhs
        T2_rhs_ins = T2_rhs()  # Make an instance of rhs

        symb_time_units = C1_rhs_ins.CheckSymbTimeUnits(C1_rhs_ins)
        # Genetare the director
        RHS_list = [C1_rhs_ins, V1_rhs_ins, T1_rhs_ins, T2_rhs_ins]
        self.MergeVarsFromRHSs(RHS_list, call=__name__)
        self.AddModule('Module1', Module1(C1=C1_rhs_ins, V1=V1_rhs_ins, T1=T1_rhs_ins, T2=T2_rhs_ins))
        self.sch = ['Module1']


    def reset(self):
        self.Vars['T1'].val = np.random.normal(21, 2)
        self.Vars['T2'].val = np.random.normal(21, 2)
        self.Vars['V1'].val = V1_in
        self.Vars['C1'].val = np.random.normal(500, 1)

    def update_controls(self, U=np.ones(10)):
        for i in range(len(U)):
            self.Vars['U'+str(i+1)].val = U[i]

    

# Attention: n must be equal to nrec in the RHSs

#Dir.Run(Dt=1, n=24*60, sch=Dir.sch)
#print(Dir.OutVar('C1'), Dir.OutVar('V1'), Dir.OutVar('T1'), Dir.OutVar('T2'))
#T1_mean = Dir.OutVar('T1')
# return Dir
#################################################################
############### Función - Lectura de datos ######################
##################################################################