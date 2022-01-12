from .struct_var import Struct
from sympy import symbols
import numpy as np
s, mol_CO2, mol_air, mol_phot, m, d, C, g, mol_O2, pa, ppm = symbols('s mol_CO2 mol_air mol_phot m d C g mol_O2 pa ppm')
mu_mol_CO2 = 1e-6 * mol_CO2
mu_mol_phot = 1e-6 * mol_phot
mu_mol_O2 = 1e-6 * mol_O2
mg = 1e-3*g
## Growth model
nmrec=115
nrec = nmrec
ok = 'ok'
n_f, n_p, MJ = symbols('n_f n_p MJ') # number of fruits, number of plants
theta = np.array([0.7, 3.3, 0.25])
S_mean = [432, 14, 20, 20]
C1M, _, _, T2M  = S_mean

STATE_VARS = {
    'Ci':Struct(typ='State', varid='Ci', prn=r'$C_i$',\
                    desc="Intracellular CO2", units= ppm , val=410, rec=nrec,ok = ok),
    'A':Struct(typ='State', varid='A', prn=r'$A$',\
           desc="Assimilation rate", units= g * (m**-2), val=0, rec=nrec,ok = ok),
       }

INPUTS = {
       'C1':Struct(typ='State', varid='C1', prn=r'$C_1$',\
           desc="CO2 concentration in the greenhouse air", \
           units= mu_mol_CO2 * mol_air**-1, val=C1M,ok = ok) ,
        
       'RH':Struct(typ='State', varid='RH', prn=r'$RH$',\
           desc="Relative humidity percentage in the greenhouse air", \
           units=1, val=50,ok = ok),
        
       'T':Struct(typ='State', varid='T', prn=r'$T$',desc="Greenhouse air temperature", units= C , val=T2M, rec=nrec,ok = ok ),
        
       'PAR':Struct(typ='State', varid='PAR', prn=r'$PAR$',\
            desc="PAR radiation", units=mu_mol_phot * (m**-2) * d**-1 , val=300.00, rec=nrec, ok = ok)
}

STOMAL_RESISTANCE = {
        'k_Ag':Struct( typ='Cnts', varid='k_Ag', \
           desc="Constant for units transformation", \
           units= m**3 * g**-1 * s**-1 * mu_mol_CO2 * mol_air**-1, val=1, ok = ok ),

        'r_m':Struct( typ='Cnts', varid='r_m', \
           desc="minimal stomatal resistance", \
           units= s * m**-1, val=100,ok = ok ),
        
        'C_ev1':Struct( typ='Cnts', varid='C_ev1', \
           desc="Constant in the formula of f_R", \
           units= mu_mol_phot * (m**-2) * d**-1, val=4.3,ok = ok ),
        
        'C_ev2':Struct( typ='Cnts', varid='C_ev2', \
           desc="Constant in the formula of f_R", \
           units= mu_mol_phot * (m**-2) * d**-1 , val=0.54, ok = ok),
        
        'k_fc':Struct( typ='Cnts', varid='k_fc', \
           desc="Constant for units completation", \
           units= mu_mol_CO2 * mol_air**-1, val=1, ok = ok ),
        
        'C_ev3d':Struct( typ='Cnts', varid='C_ev3d', \
           desc="Constant in the formula of f_C", \
           units= mol_air * mu_mol_CO2**-1, val=6.1e-7, ok = ok ),
        
        'C_ev3n':Struct( typ='Cnts', varid='C_ev3n', \
           desc="Constant in the formula of f_C", \
           units= mol_air * mu_mol_CO2**-1, val=1.1e-11, ok = ok),
        
        'S':Struct( typ='Cnts', varid='S', \
           desc="Constant in the formula of Sr", \
           units= m**2 * d * mu_mol_phot**-1, val=-1,ok = ok),
        
        'Rs':Struct( typ='Cnts', varid='Rs', \
           desc="Radiation setpoint to switch day and night", \
           units= mu_mol_phot * (m**-2) * d**-1, val=5, ok = ok),
        
        'C_ev4d':Struct( typ='Cnts', varid='C_ev4d', \
           desc="Constant in the formula of f_C", \
           units= pa**-1, val=4.3e-6, ok = ok ),
        
        'C_ev4n':Struct( typ='Cnts', varid='C_ev4n', \
           desc="Constant in the formula of f_C", \
           units= pa**-1, val=5.2e-6)
}

CO2_ABSORPTION = {
        'ks' : Struct( typ='Cnts', varid='ks', \
           desc="Stomatal ratio", \
           units= 1, val=0.5, ok = ok ),
        
        'Rb':Struct( typ='Cnts', varid='Rb', \
           desc="Stomatal resistance of the canopy boundary layer", \
           units= s * m**-1, val=711, ok = ok )
}
        
ASSIMILATES={
        'k_d':Struct( typ='Cnts', varid='k_d', \
           desc="factor to transform s**-1 into d**-1", units=1, val=1),
        
        'k_T':Struct( typ='Cnts', varid='k_T', \
           desc="Auxiliary constant to add temperature units", units= C, val=1.0),
        
        'k_JV':Struct( typ='Cnts', varid='k_JV', \
           desc="Auxiliary constant which transforms the units of the electron transport rate, J to those of the maximum Rubisco rate, V_cmax", \
           units= mu_mol_CO2 * mu_mol_phot**-1, val=1.0),
        
        'fc':Struct( typ='Cnts', varid='fc', \
           desc="Factor to transform mu-mols_CO2/sec to grms_CH20/day", \
           units= g * d * mu_mol_CO2**-1 , val=3.418181e-1), # 7.891414141414142e-6
        
        'phi':Struct( typ='Cnts', varid='phi', \
           desc="Ratio of oxigenation to carboxylation rates", \
           units= mu_mol_O2 * mu_mol_CO2**-1, val=2),
        
        'O_a':Struct( typ='Cnts', varid='O_a', \
           desc="O2 concentration in the enviroment", \
           units= mu_mol_O2 * mol_air**-1, val=210000),
        
        'V_cmax25':Struct( typ='Cnts', varid='V_cmax25', \
           desc="Maximum Rubisco Rate, per unit area", \
           units= mu_mol_CO2 * (m**-2) * d**-1, val=200),
        
        'Q10_Vcmax':Struct( typ='Cnts', varid='Q10_Vcmax', \
           desc="Temperatura response of Vcmax", \
           units=1, val=2.4),
        
        'K_C25':Struct( typ='Cnts', varid='K_C25', \
           desc="Michaelis-Menten for CO2", \
           units= mu_mol_CO2 * mol_air**-1 , val=300),
        
        'Q10_KC':Struct( typ='Cnts', varid='Q10_KC', \
           desc="Temperatura response of Michaelis-Menten for CO2", \
           units=1, val=2.1),
        
        'K_O25':Struct( typ='Cnts', varid='K_O25', \
           desc="Michaelis-Menten for O2", \
           units= mu_mol_O2 * mol_air**-1 , val=3e5),
        
        'Q10_KO':Struct( typ='Cnts', varid='Q10_KO', \
           desc="Temperatura response of Michaelis-Menten for O2", \
           units=1, val=1.2) ,
        
        'tau_25':Struct( typ='Cnts', varid='tau_25', \
           desc="Specificity factor", \
           units=1 , val=2600),
        
        'Q10_tau':Struct( typ='Cnts', varid='Q10_tau', \
           desc="Temperatura response of specificity factor", \
           units=1, val=2.1) ,
        
        'J_max':Struct( typ='Cnts', varid='J_max', \
           desc="Maximum electron transport rate", \
           units= mu_mol_phot * (m**-2) * d**-1, val=400),
        
        'ab':Struct( typ='Cnts', varid='ab', \
           desc="Leafs absorbance", \
           units=1 , val=0.85),
        
        'f':Struct( typ='Cnts', varid='f', \
           desc="Correction factor for the spectral quality of the light", \
           units=1 , val=0.15),
        
        'theta':Struct( typ='Cnts', varid='theta', \
           desc="Empirical factor", \
           units=1 , val=theta[0])
}