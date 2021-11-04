from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, CONTROLS, COSTS, FUNCTIONS


mt = symbols('mt')


class Qco2_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qco2, the co2 cost per m^2"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        COSTS['Qco2'].addvar_rhs(self)
        CONTROLS['U10'].addvar_rhs(self)
        CONSTANTS['psi2'].addvar_rhs(self)
        CONSTANTS['alpha6'].addvar_rhs(self)
        CONSTANTS['q_co2_ext'].addvar_rhs(self)
        FUNCTIONS['o2'].addvar_rhs(self)
    
    def RHS(self, Dt):
        '''Costo del CO_2'''
        o_2 = self.V('o2')
        return (10**-6)*self.V('q_co2_ext')*o_2