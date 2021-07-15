from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, BETA, GAMMA, EPSIL, LAMB, PHI
from .constants import CONSTANTS, INPUTS, INIT_STATE, nmrec
from .functions import a1
from .functions import g1
from .functions import r6
from .functions import h4, h6
from .functions import H_Boil_Pipe


qgas, etagas, _, _, sigma = CONSTANTS.values()
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()
C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, mxn= symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm mxn')


class Qgas_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qgas, the gas cost per m^2"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='Qgas', prn=r'$Q_{Gas}$',
                    desc="Fuel cost (natural gas)", units=mxn * m**-2, val=0, rec=nrec)
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        # Inputs
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=I3)
        # Constants 
        self.AddVar(typ='Cnts', varid='beta3', prn=r'$\beta_3$',
                    desc="Canopy extinction coefficient for NIR radiation", units=1, val=BETA['beta3']) # ok
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=LAMB['lamb4'])  # Dr Antonio dio el valor
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=ALPHA['alpha6'])  # ok
        self.AddVar(typ='Cnts', varid='alpha3', prn=r'$\alpha_3$',
                    desc="Surface of the heating pipe", units=m**2*m**-2, val=ALPHA['alpha3']) # Valor proporcionado por Dr Antonio
        self.AddVar(typ='Cnts', varid='lamb', prn=r'$\lambda$',
                    desc="Boltzmann constant", units=W * m**-2 * K**-4, val=sigma) # ok
        self.AddVar(typ='Cnts', varid='epsil2', prn=r'$\epsilon_2$',
                    desc="Canopy FIR emission coefficient", units=1, val=EPSIL['epsil2']) # ok
        self.AddVar(typ='Cnts', varid='epsil1', prn=r'$\epsilon_1$',
                    desc="FIR emission coefficient of the heating pipe", units=1, val=EPSIL['epsil1']) # ok
        self.AddVar(typ='Cnts', varid='gamma1', prn=r'$\gamma_1$',
                    desc="Length of the heating pipe", units=m * m**-2, val=GAMMA['gamma1']) # ok ---> Usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi1', prn=r'$\phi_1$',
                    desc="External diameter of the heating pipe", units=m, val=PHI['phi1']) # ok
        self.AddVar(typ='Cnts', varid='etagas', prn=r'$\eta_{gas}$',
                    desc="Energy efficiency of natural gas", units=1, val=etagas) # checar unidades
        self.AddVar(typ='Cnts', varid='qgas', prn=r'$q_{gas}$',
                    desc="Cost of natural gas", units=1, val=qgas) # checar unidades
        # falta agregar constantes qgas, etagas

    def RHS(self, Dt):
        h_6 = h6(U4=self.V('U4'), lamb4=self.V('lamb4'), alpha6=self.V('alpha6')) #H blow air 
        a_1 = a1(I1=self.V('I1'), beta3=self.V('beta3')) #auxiliar para g1
        g_1 = g1(a1=a_1)                                   #auxiliar para r6
        r_6 = r6(T1=self.V('T1'), I3=self.V('I3'), alpha3=self.V('alpha3'), epsil1=self.V('epsil1'), epsil2=self.V('epsil2'), lamb=self.V('lamb'), g1=g_1)
        h_4 = h4(T2=self.V('T2'), I3=self.V('I3'),gamma1=self.V('gamma1'), phi1=self.V('phi1'))
        H_boil_pipe = H_Boil_Pipe(r_6, h_4)
        return (self.V('qgas')/self.V('etagas'))*(H_boil_pipe + h_6)/(10**9)


