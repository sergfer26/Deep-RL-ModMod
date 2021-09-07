from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, INPUTS, STATE_VARS, CONTROLS, COSTS, FUNCTIONS
from .functions import H_Boil_Pipe
from .functions import h4, h6
from .functions import a1
from .functions import g1
from .functions import r6


mt= symbols('mt')

constant_names = ['beta3', 'lamb4', 'alpha6', 'alpha3', 'sigma', 'epsil2', 'epsil1', 
                            'gamma1', 'phi1', 'etagas', 'qgas']

function_names = ['h6', 'r6', 'h4', 'a1', 'g1'] 



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

        for name in function_names:
            FUNCTIONS[name].addvar_rhs(self)

    def RHS(self, Dt):
        h_6 = self.V('h6')
        r_6 = self.V('r6')
        h_4 = self.V('h4')
        H_boil_pipe = H_Boil_Pipe(r_6, h_4)
        return (self.V('qgas')/self.V('etagas'))*(H_boil_pipe + h_6)/(10**9)


