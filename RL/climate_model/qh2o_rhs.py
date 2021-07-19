from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, INPUTS, STATE_VARS, CONTROLS, COSTS
from .functions import q1, q2, q3, q4, q5, q6, q7, q8, q9, q10
from .functions import p1, p2, p3
from .functions import kappa3
from .functions import f1
from .functions import h3


mt = symbols('mt')
state_names = ['T1', 'V1', 'T2', 'C1']
control_names = ['U2', 'U3', 'U9']
input_names = ['I1', 'I6', 'I14']
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
        
    def RHS(self, Dt):
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        q_2 = q2(T1=self.Vk('T1'))
        q_7 = q7(I14=self.V('I14'), delta1=self.V(
            'delta1'), gamma5=self.V('gamma5'))
        q_8 = q8(delta4=self.V('delta4'), delta5=self.V('delta5'), q7=q_7)
        q_9 = q9(delta6=self.V('delta6'), delta7=self.V('delta7'), q7=q_7)
        q_4 = q4(C1=self.V('C1'), eta4=self.V('eta4'), q8=q_8)
        q_5 = q5(V1=self.V('V1'), q2=q_2, q9=q_9)
        q_6 = q6(I6=self.V('I6'))
        q_10 = q10(I14=self.V('I14'), delta2=self.V(
            'delta2'), delta3=self.V('delta3'))
        q_3 = q3(I14=self.V('I14'), gamma4=self.V(
            'gamma4'), q4=q_4, q5=q_5, q10=q_10)
        q_1 = q1(I1=self.V('I1'), rho3=self.V('rho3'), alpha5=self.V('alpha5'), gamma=self.V(
            'gamma'), gamma2=self.V('gamma2'), gamma3=self.V('gamma3'), q3=q_3)
        h_3 = h3(T2=self.V('T2'), V1=self.V('V1'), U3=self.V('U3'), I6=self.V('I6'), lamb1=self.V(
            'lamb1'), lamb2=self.V('lamb2'), alpha6=self.V('alpha6'), gamma2=self.V('gamma2'), q6=q_6)
        #### Principal functions ####
        kappa_3 = kappa3(T2=self.Vk('T2'), psi1=self.V(
            'psi1'), phi2=self.V('phi2'), omega2=self.V('omega2'))
        p_1 = p1(V1=self.Vk('V1'), q1=q_1, q2=q_2)
        
        p_2 = p2(rho3=self.V('rho3'), eta5=self.V('eta5'),
                    phi5=self.V('phi5'), phi6=self.V('phi6'), f1=f_1)
        p_3 = p3(U9=self.V('U9'), phi9=self.V(
            'phi9'), alpha6=self.V('alpha6'))
        return (10**-3)*((1+ self.V('etadrain')/100.0)*max(p_1, 0) + p_2 + p_3)