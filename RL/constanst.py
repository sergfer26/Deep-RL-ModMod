from struct_var import Struct
from sympy import symbols

mt, mg, m, C, s, W, mg_CO2, J, g, mol_CH2O = symbols('mt mg m C s W mg_CO2 J g mol_CH2O')

mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, m_cover, kg_air = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm m_cover kg_air')  # Symbolic use of base phisical units

mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, kmol, kg_air, kg_vapour, mxn = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm kmol kg_air kg_vapour mxn')  # Symbolic use of base phisical units
ok = 'OK'
RHO = {
    'rho1':Struct(typ='Cnts', varid='rho1', prn=r'$\rho_1$',
                    desc="PAR reflection coefficient", units=1, val=0.07,ok=ok),
    'rho2': Struct(typ='Cnts', varid='rho2', prn=r'$\rho_2$',
                    desc="Floor reflection coefficient PAR", units=1, val=0.65,ok = ok), 
    'rho3': Struct(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= 1.2,ok = 'El valor es el de la densidad del aire al nivel del mar'),
    'rho4': Struct(desc='Densidad del aire a nivel del mar')
}


TAU = {
    'tau1': Struct(typ='Cnts', varid='tau1', prn=r'$\tau_1$',
                    desc="PAR transmission coefficient of the Cover", units=1, val=1,ok = 'En el artículo no dan su valor'),
    'tau2': Struct(typ='Cnts', varid='tau2', prn=r'$\tau_2$',
                    desc="FIR transmission coefficient of the Cover", units=1, val=1, ok ='En el artículo no dan su valor'),
    'tau3': Struct(typ='Cnts', varid='tau3', prn=r'$\tau_3$',
                    desc="FIR transmission coefficient of the thermal screen", units=1, val=0.11,ok = 'ok --> usé el valor de Texas') 
}


NU = {
    'nu1': Struct(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=0.65,ok = ok), 
    'nu2': Struct(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=0.1,ok=ok),
    'nu3': Struct(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=0,ok  = 'En ejemplos del artículo usan valor cero'), 
    'nu4': Struct(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=1e-4,ok=ok), 
    'nu5': Struct(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=2e3, ok = ' 0.2*alpha6 --> ok'), 
    'nu6': Struct(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=1,ok=ok), 
    'nu7': Struct(typ='Cnts', varid='nu7', prn=r'$\nu_7$',
                    desc="Soil thermal conductivity", units=W * m**-1 * K**-1, val=0.85, ok=ok), 
    'nu8': Struct(typ='Cnts', varid='nu8', prn=r'$\nu_8$',
                    desc="Floor to ground distance", units=m, val=0.64,ok=ok) 
}


PHI = {
    'phi1': Struct(typ='Cnts', varid='phi1', prn=r'$\phi_1$',
                    desc="External diameter of the heating pipe", units=m, val=51e-3,ok = ok),
    'phi2': Struct(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=20, ok = 'Valor original 4'), 
    'phi3': Struct(desc='Masa molar del aire'), 
    'phi4': Struct('Altitud del invernadero'),
    'phi5': Struct(typ='Cnts', varid='phi5', prn=r'$\phi_5$',
                    desc="Water vapor contained in the fan-pad system", units=kg_water * kg_air**-1, val=0,ok = 'Falta valor --> En realaidad es un input'), 
    'phi6': Struct(typ='Cnts', varid='phi6', prn=r'$\phi_6$',
                    desc="Water vapor contained in the outside air", units=kg_water * kg_air**-1, val=0, ok = 'Falta valor --> En realaidad es un input'), # Vapor de agua contenido en el aire exterior
    'phi7': Struct(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=16.7,ok = ok), 
    'phi8': Struct(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=0, ok = 'Falta valor, aunque en los ejemplos del artículo no se considera'),
    'phi9': Struct(typ='Cnts', varid='phi9', prn=r'$\phi_9$',
                    desc="Fog system capacity", units=kg_water * s**-1, val=0,ok = 'Falta valor'), 
}


PSI = {
    'psi1':Struct(typ='Cnts', varid='psi1', prn=r'$\psi_1$',
                    desc="Molar mass of water", units=kg * kmol**-1, val=18,ok = ok), 
    'psi2': Struct(typ='Cnts', varid='psi2', prn=r'$\psi_2$',
                    desc="Capacity of the external CO2 source", units=mg * s**-1, val=7.2*(10**4),ok = 'Tenia el valor de Texas'),
    'psi3': Struct(typ='Cnts', varid='psi3', prn=r'$\psi_3$',
                    desc="Molar mass of the CH2O", units=g * mol_CH2O**-1, val=30.031,ok = ok)
}


OMEGA = {
    'omega1': Struct(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=9.81, ok = ok), 
    'omega2': Struct(typ='Cnts', varid='omega2', prn=r'$\omega_2$',
                    desc="Molar gas constant", units=J * kmol**-1 * K**-1, val= 8.314e3, ok = ok), 
    'omega3': Struct(typ='Cnts', varid='omega3', prn=r'$\omega_3$',\
                        desc="Percentage of CO2 absorbed by the canopy", units= 1 , val=0.03,ok = 'Sin comentario') 
}


################## Inputs ##################
INPUTS = {
    'I1' : Struct(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=3, ok = 'Valor tomado de internet'),
    'I2' : Struct(typ='State', varid='I2', prn=r'$I_2$',
                    desc="External global radiation", units=W * m**-2, val=100.0, ok = 'Sin comentario'), 
    'I3' : Struct(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=20, ok = 'Sin comentario'),      
    'I4' : Struct(typ='State', varid='I4', prn=r'$I_4$',
                    desc="Sky temperature", units=C, val=0,ok = 'Sin comentario'),      
    'I5' : Struct(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=18, ok = 'Sin comentario'),  
    'I6' : Struct(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val = 20, ok = 'Sin comentario'),      # Mechanical cooling system temperature 
    'I7' : Struct(typ='Cnts', varid='I7', prn=r'$I_7$',
                    desc="Soil temperature", units=C, val=5, ok = 'Valor tomado de internet'),  
    'I8' : Struct(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=3.2, ok = 'Sin comentario'),     
    'I9' : Struct(),
    'I10' : Struct(typ='Cnts', varid='I10', prn=r'$I_{10}$',
                    desc="Outdoor CO2 concentration", units=mg * m**-3, val = 700,ok = 'Sin comentario'),    
    'I11' : Struct(typ='Cnts', varid='I11', prn=r'$I_{11}$',
                    desc="Inhibition of the rate of photosynthesis by saturation of the leaves with carbohydrates", units=1, val=0, ok = 'Falta valor y unidades'),     
    'I12' : Struct(typ='Cnts', varid='I12', prn=r'$I_{12}$',
                    desc="Crude canopy photosynthesis rate", units=1, val=0,ok = 'Falta valor y unidades'),     
    'I13' : Struct(typ='Cnts', varid='I13', prn=r'$I_{13}$',
                    desc="Photorespiration during photosynthesis", units=1, val=0,ok = 'Falta valor y unidades'),      # FALTA VALOE Photorespiration during photosynthesis
    'I14' : Struct(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=100, ok = ' Sin comentario')   
}
