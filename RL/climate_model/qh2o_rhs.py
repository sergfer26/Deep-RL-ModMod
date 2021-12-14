from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, FUNCTIONS, INPUTS, STATE_VARS, CONTROLS, COSTS
from .functions import q1, q2, q3, q4, q5, q7, q8, q9, q10
from .functions import p1, p2, p3
from .functions import f1


mt = symbols('mt')
state_names = ['T1', 'V1', 'T2', 'C1']
control_names = ['U2', 'U3', 'U9']
input_names = ['I1', 'I6']
function_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'q4', 'q5', 'q7', 'q8', 'q9', 'q10', 'f1']
constant_names = ['alpha6', 'phi7', 'lamb1', 'lamb2', 'gamma2', 'rho3', 'alpha5', 
                            'gamma', 'gamma3', 'delta1', 'delta2', 'delta3', 'gamma4', 
                            'gamma5', 'delta4', 'delta5', 'delta6', 'delta7', 'eta4',
                            'psi1', 'phi2', 'omega2', 'eta5', 'phi5', 'phi6', 'phi9',
                            'etadrain']


class Qh2o_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qh20, the water cost per kg"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        COSTS['Qh2o'].addvar_rhs(self)
        for name in state_names:
            STATE_VARS[name].addvar_rhs(self)
        
        for name in control_names:
            CONTROLS[name].addvar_rhs(self)

        for name in input_names:
            INPUTS[name].addvar_rhs(self)

        for name in constant_names:
            CONSTANTS[name].addvar_rhs(self)

        for name in function_names:
            FUNCTIONS[name].addvar_rhs(self)
        
    def RHS(self, Dt):
        p_1 = self.V('p1')
        p_2 = self.V('p2')
        p_3 = self.V('p3')
        return (10**-3)*((1+ self.V('etadrain')/100.0)*max(p_1, 0) + p_2 + p_3)