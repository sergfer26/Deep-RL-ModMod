from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA
from .constants import CONSTANTS, INPUTS, INIT_STATE, nmrec, theta
from .functions import o2


lamb4, alpha6, phi1, gamma1, alpha3, psi2, qgas, etagas, q_co2_ext, T_cal = CONSTANTS.values()
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()
C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, mxn= symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm mxn')


class Qco2_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qco2, the co2 cost per m^2"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='Qco2', prn=r'$Q_{Gas}$',
                    desc="Fuel cost from external source", units=mxn * m**-2, val=0, rec=nrec)
        # control variables  ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U10', prn=r'$U_{10}$',
                    desc="Control of external CO2 source", units=1, val=0)
        # Constants 
        self.AddVar(typ='Cnts', varid='psi2', prn=r'$\psi_2$',
                    desc="Capacity of the external CO2 source", units=mg * s**-1, val=theta[2])  # Falta valor, tomé el del ejemplo de Texas 4.3e5
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=ALPHA['alpha6']) # ok
    
    def RHS(self, Dt):
        '''Costo del CO_2'''
        o_2 = o2(U10=self.V('U10'), psi2=self.V('psi2'), alpha6=self.V('alpha6')) #MC_ext_air
        return (10**6)*q_co2_ext*o_2