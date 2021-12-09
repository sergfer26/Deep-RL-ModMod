from ModMod import StateRHS
from sympy import symbols
from .constants import CONSTANTS, INPUTS, STATE_VARS, CONTROLS, FUNCTIONS
from .functions import f1, f2, f3, f4, f5, f6, f7
from .functions import o1, o2, o3, o4, o5, o6 
from .functions import h6, n1, n2, n3
from .functions import kappa4
from .functions import Amg

# Symbolic use of base phisical units
mt = symbols('mt')

state_names = ['T2', 'C1']
control_names = ['U1', 'U2', 'U4', 'U5', 'U6', 'U7', 'U8', 'U10']
input_names = ['I8', 'I5', 'I10', 'I11']
function_names = ['f1', 'h6', 'o2']
constant_names = ['lamb4', 'alpha6', 'phi7', 'eta6', 'eta7', 'eta8', 'phi8', 'nu4', 
                            'nu5', 'omega1', 'nu6', 'nu1', 'eta10', 'nu3', 'nu2', 'eta11', 
                            'phi2', 'eta13', 'psi2', 'psi3', 'omega3']

class C1_rhs(StateRHS):
    """Define a RHS, this is the rhs for C1, the CO2 concentrartion in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
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
        """RHS( Dt, k) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
        """
        # Direct usage, NB: State variables need to used Vk, so that X+k is evaluated.
        # This can be done with TranslateArgNames(h1)
        # Once defined h1 in your terminal run TranslateArgNames(h1)
        # and follow the instrucions
        #### Sub-functions ####
        h_6 = self.V('h6')
        f_1 = self.V('f1')
        o_2 = self.V('o2')
        f_3 = f3(U7=self.V('U7'), phi8=self.V('phi8'), alpha6=self.V('alpha6'))
        f_6 = f6(I8=self.V('I8'), nu4=self.V('nu4'))
        n_1 = n1(U5=self.V('U5'), nu1=self.V('nu1'), eta10=self.V('eta10'))
        n_2 = n2(U6=self.V('U6'), nu3=self.V('nu3'))
        n_3 = n3(U5=self.V('U5'), nu2=self.V('nu2'), eta11=self.V('eta11'))
        f_5 = f5(I8=self.V('I8'), alpha6=self.V(
            'alpha6'), n1=n_1, n2=n_2, n3=n_3)
        f_2 = f2(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f5=f_5, f6=f_6)
        f_7 = f7(T2=self.Vk('T2'), U8=self.V('U8'), I5=self.V('I5'), I8=self.V('I8'), nu5=self.V(
            'nu5'), alpha6=self.V('alpha6'), omega1=self.V('omega1'), nu6=self.V('nu6'), n1=n_1, n3=n_3)
        f_4 = f4(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f6=f_6, f7=f_7)
        #### Principal functions ####
        kappa_4 = kappa4(phi2=self.V('phi2'))
        o_1 = o1(eta13=self.V('eta13'), h6=h_6)
        o_3 = o3(C1=self.Vk('C1'), I10=self.V('I10'), f1=f_1)
        o_4 = Amg(C=self.Vk('C1'),PAR = self.V('I2'))
        o_5 = o5(C1=self.Vk('C1'), I10=self.V(
            'I10'), f2=f_2, f3=f_3, f4=f_4)
        return (kappa_4**-1)*(o_1 + o_2 + o_3 - o_4 - o_5 )