from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, INPUTS, STATE_VARS, CONTROLS, COSTS
from .functions import H_Boil_Pipe
from .functions import h4, h6
from .functions import a1
from .functions import g1
from .functions import r6


mt= symbols('mt')

constant_names = ['beta3', 'lamb4', 'alpha6', 'alpha3', 'sigma', 'epsil2', 'epsil1', 
                            'gamma1', 'phi1', 'etagas', 'qgas']



class Qgas_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qgas, the gas cost per m^2"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        COSTS['Qgas'].addvar_rhs(self)
        CONTROLS['U4'].addvar_rhs(self)
        CONTROLS['U11'].addvar_rhs(self)
        INPUTS['I1'].addvar_rhs(self)
        INPUTS['I3'].addvar_rhs(self)
        STATE_VARS['T1'].addvar_rhs(self)
        STATE_VARS['T2'].addvar_rhs(self)
        for name in constant_names:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        h_6 = h6(U4=self.V('U4'), lamb4=self.V('lamb4'), alpha6=self.V('alpha6')) #H blow air 
        a_1 = a1(I1=self.V('I1'), beta3=self.V('beta3')) #auxiliar para g1
        g_1 = g1(a1=a_1)                                   #auxiliar para r6
        r_6 = r6(T1=self.V('T1'), I3=self.V('I3'), alpha3=self.V('alpha3'), epsil1=self.V('epsil1'), epsil2=self.V('epsil2'), lamb=self.V('sigma'), g1=g_1)
        h_4 = h4(T2=self.V('T2'), I3=self.V('I3'),gamma1=self.V('gamma1'), phi1=self.V('phi1'))
        H_boil_pipe = H_Boil_Pipe(r_6, h_4)
        return (self.V('qgas')/self.V('etagas'))*(H_boil_pipe + h_6)/(10**9)


