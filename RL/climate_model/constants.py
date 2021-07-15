import numpy as np
# from .constants import ALPHA, BETA, GAMMA, DELTA, EPSIL, ETA, LAMB, RHO, TAU, NU, PHI, PSI, OMEGA


################## Constants ##################
CONSTANTS = {      
    'qgas': 2.45,         # Precio del gas natural $(mxn) m^-3 
    'etagas': 35.26,      # Eficiencia energ ́etica del gas natural
    'q_co2_ext': 3.5,     # Costo del gas de la fuente externa lo tomamos al precio de la tesis 
    'T_cal': 95,          # Temperatura máxima de la caldera  
    'sigma': 5.670*(10**−8), # Constante de Stefan-Boltzmann (W m−2)
    'etadrain' : 30
}


ALPHA = {
    'alpha1': 0, # Capacidad calórifica de un m^2 de dosel (theta[0])
    'alpha2': 0.35, # Coeficiente global de absorción NIR del dosel
    'alpha3': 0.3, # Superficiedelatuber ́ıadecalentamiento
    'alpha4': 5, # Coeficiente de intercambio de calor por conveccio ́n de la hoja del dosel al aire del invernadero
    'alpha5': 1e3, # Capacidadcalor ́ıficaespecificadelaire del invernadero
    'alpha6': 10**4, # Área de la superficie del piso del invernadero
    'alpha7': 0.5, # Coeficiente global de absorcio ́n NIR del piso
    'alpha8': 1, # Coeficiente de absorci ́on PAR de la cubierta # En el artículo no dan el valor
    'alpha9': 1  # Coeficiente de absorci ́on NIR de la cubierta # En el artículo no dan el valor
}


BETA = {
    'beta1': 0.7, # Coeficiente de extinci ́on del dosel para radiacio ́n PAR # ok
    'beta2': 0.7, # Coeficiente de extinci ́on para radiaci ́on PAR que se refleja desde el piso hasta el dosel # ok
    'beta3': 0.27, # Coeficiente de extinci ́on del dosel para radiacio ́n NIR # ok
}


GAMMA = {
    'gamma': 65.8,  # Constante psicrom ́etrica #ok 
    'gamma1': 1.875, # Longitud de la tubería de calentamiento (Almería)
    'gamma2': 2.45e6, # Calor latente de evaporaci ́on del agua #ok
    'gamma3': 275, # Resistencia de la capa l ́ımite del dosel para transporte de vapor # ok
    'gamma4': 82.0, # Resistenciaestom ́aticam ́ınimadeldosel # ok
    'gamma5': -1  # Pendiente del intercambio diferenciable para el modelo de resistencia estom ́atica #ok
}


DELTA = {
    'delta1': 5, # Radiación por encima del dosel que define el amanecer y la puesta de sol # ok
    'delta2': 4.3, # Parámetro determinado empíricamente # ok
    'delta3': 0.54, # Parámetro determinado empíricamente # ok
    'delta4': 6.1e-7, # Parámetro de transpiración del CO en el día # ok
    'delta5': 1.1e-11, # Parámetro de transpiración del CO en la noche # ok
    'delta6': 4.3e-6, # Parámetro de transpiración del vapor en el día # ok
    'delta7': 5.2e-6  # Parámetro de transpiración del vapor en la noche #ok 
}

EPSIL = {
    'epsil1': 0.88, # Coeficiente de emisión FIR de la tubería de calentamiento # ok 
    'epsil2': 1, # Coeficiente de emisión FIR del dosel # ok
    'epsil3': 1, # Coeficiente de emisión FIR del cielo # ok
    'epsil4': 1, # Coeficiente de emisión FIR del piso # ok
    'epsil5': 1, # Coeficiente de emisión FIR de la pantalla térmica
    'epsil6': 0.44  # Coeficiente de emisión FIR de la cubierta externa #ok ---> use el valor de Texas
}


ETA = {
    'eta1': 0.1,  # Proporción de la radiación global que es absorbida por los elementos de construcción del invernadero # ok
    'eta2': 1.0,  # Razón entre la radiación PAR y la radiación global externa ¿0.5?
    'eta3': 0.5,  # Razón entre la radiación NIR y la radiación global externa # ok 
    'eta4': 0.554,  # Factor de conversión de mg m−3 CO2 a ppm # ok 
    'eta5': 0.5,  # Eficiencia del sistema de ventilador-almohadilla # no da el valor en el articulo
    'eta6': 1,  # Factor de reduccio ́n de la potencia de ventilación # Falta valor
    'eta7': 0.5,  # Razón entre el área de ventilación en el techo y el  área de ventilación total  # no da el valor en el articulo
    'eta8': 0.9,  # Razón entre el área de ventilación techo y total, si no hay efecto de chimenea # ok
    'eta9': 0,  # Razón entre el área de ventilación lateral y el área de ventilación total # no hay eta9
    'eta10': 0, # Efecto de la sombra sobre el coeficiente de descarga # Falta valor
    'eta11': 0, # Efecto de la sombra sobre el coeficiente de presión global del viento # Falta valor, aunque en los ejemplos del artículo no se considera
    'eta12': 4.43e-8, # Cantidad de vapor que es liberado cuando un joule de energía sensible es producido por el calentador de aire directo # ok
    'eta13': 0.057  # Cantidad de CO2 que es liberado cuando un joule de energía sensible es producido por el calentador de aire directo # ok
}


LAMB = {
    'lamb1': 0, # Coeficiente de desempen ̃o del sistema de enfriamiento meca ́nico # Falta valor, aunque en los ejemplos del artículo no se considera
    'lamb2': 0, # Capacidad el ́ectrica del sistema de enfriamiento meca ́nico # Falta valor, aunque en los ejemplos del artículo no se considera
    'lamb3': 0, # Coeficiente de intercambio de calor convictivo entre el suelo y el aire del invernadero # Falta valor, aunque en los ejemplos del artículo no se considera
    'lamb4': 5*(10**5), # Capacidad calor ́ıfica del calentador de aire directo
    'lamb5': 1.8e4, # Superficie de la cubierta # ok --> tomé el valor de Holanda, el de Texas es muy grande (9e4)
    'lamb6': 2.8, # Variable de intercambio de calor por convecci ́on entre la cubierta y el aire exterior # ok ---> usé el valor de Texas
    'lamb7': 1.2, # Variable de intercambio de calor por convecci ́on entre la cubierta y el aire exterior # ok ---> usé el valor de Texas
    'lamb8': 1 # Variable de intercambio de calor por convecci ́on entre la cubierta y el aire exterior # ok ---> usé el valor de Texas
}


RHO = {
    'rho1': 0, # Coeficiente de reflexi ́on PAR
    'rho2': 0, # Coeficiente de reflexi ́on PAR del piso
    'rho3': 0, # Densidad del aire
    'rho4': 0  # Densidad del aire a nivel del mar
}


TAU = {
    'tau1': 0, # Coeficiente de transmisi ́on PAR de la cubierta
    'tau2': 0, # Coeficiente de transmisi ́on FIR de la cubierta
    'tau3': 0  # Coeficiente de transmisi ́on FIR de la pantalla t ́ermica
}


NU = {
    'nu1': 0, # Coeficiente de descarga sin sombra
    'nu2': 0, # Coeficiente de la presio ́n global del viento sin sombra
    'nu3': 0, # Superficie lateral del invernadero
    'nu4': 0, # Coeficiente de fuga
    'nu5': 0, # Ma ́xima a ́rea de ventilaci ́on del techo
    'nu6': 0, # Dimensio ́n vertical de un s ́olo respirador abierto
    'nu7': 0, # Conductividad t ́ermica del suelo
    'nu8': 0  # Distancia del piso al suelo
}


PHI = {
    'phi1': 51e-3,, # Dia ́metro externo de la tuber ́ıa de calentamiento
    'phi2': 0, # Altura media del aire del invernadero
    'phi3': 0, # Masa molar del aire
    'phi4': 0, # Altitud del invernadero
    'phi5': 0.014, # Vapor de agua contenido en el sistema de ventilador-almohadilla
    'phi6': 0, # Vapor de agua contenido en el aire exterior
    'phi7': 0, # Capacidad del flujo de aire a trav ́es de la almohadilla
    'phi8': 0, # Capacidad de flujo de aire del sistema de ventilacio ́n forzada
    'phi9': 0, # Capacidad del sistema de niebla
}


PSI = {
    'psi1': 0, # Masa molar del agua
    'psi2': 7.2*(10**4), # Capacidad de la fuente externa de CO2
    'psi3': 0  # Masa molar del CH2O
}


OMEGA = {
    'omega1': 0, # Constante de la aceleracio ́n de la gravedad
    'omega2': 0, # Constante molar del gas
    'omega3': 0, # Percentage of CO2 absorbed by the canopy
}


################## Inputs ##################
INPUTS = {
    'I1' : 3.0,     # Leaf area index
    'I2' : 100.0,   # External global radiation
    'I3' : 20,      # Heating pipe temperature
    'I4' : 0,       # Sky temperature
    'I5' : 18.0,    # Outdoor temperature
    'I6' : 20,      # Mechanical cooling system temperature 
    'I7' : 5.0,     # Soil temperature
    'I8' : 3.2,     # Outdoor wind speed
    # I9
    'I10' : 700,    # Outdoor CO2 concentration
    'I11' : 0,      # FALTA VALOR Inhibition of the rate of photosynthesis by saturation of the leaves with carbohydrates 
    'I12' : 0,      # FALTA VALOR Crude canopy photosynthesis rate
    'I13' : 0,      # FALTA VALOE Photorespiration during photosynthesis
    'I14' : 100.0   # Global radiation above the canopy
}


INIT_STATE = {
    'C1_in' : 432,
    'V1_in' : 14,
    'T1_in' : 20,
    'T2_in' : 20
}

theta = np.array([3000, 20, 7.2*(10**4)]) # psi2 = 7.2*(10**4)
nmrec = 1