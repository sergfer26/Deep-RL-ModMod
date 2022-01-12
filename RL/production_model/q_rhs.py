from ModMod import Module, StateRHS
import numpy as np
from sympy import symbols
S_mean = [432, 14, 20, 20]
nmrec=115
theta = np.array([0.7, 3.3, 0.25])

s, mol_CO2, mol_air, mol_phot, m, d, C, g, mol_O2, pa, ppm = symbols('s mol_CO2 mol_air mol_phot m d C g mol_O2 pa ppm')
mu_mol_CO2 = 1e-6 * mol_CO2
mu_mol_phot = 1e-6 * mol_phot
mu_mol_O2 = 1e-6 * mol_O2
mg = 1e-3*g

## Growth model
n_f, n_p, MJ = symbols('n_f n_p MJ') # number of fruits, number of plants

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
