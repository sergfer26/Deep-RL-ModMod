from ModMod import Module, StateRHS, Director, ReadModule
from sympy import symbols

from .constants import ASSIMILATES, CO2_ABSORPTION, INPUTS, STATE_VARS, STOMAL_RESISTANCE
from .functions import*
import numpy as np
nmrec = 115 #oJO deberia venir de los parametros
s, mol_CO2, mol_air, mol_phot, m, d, C, g, mol_O2, pa, ppm = symbols('s mol_CO2 mol_air mol_phot m d C g mol_O2 pa ppm')

mu_mol_CO2 = 1e-6 * mol_CO2
mu_mol_phot = 1e-6 * mol_phot
mu_mol_O2 = 1e-6 * mol_O2
mg = 1e-3*g

S_mean = [432, 14, 20, 20]
C1M, _, _, T2M  = S_mean
theta = np.array([0.7, 3.3, 0.25])
class Ci_rhs(StateRHS):
    """
    Ci es el CO2 intracelular 
    """
    def __init__( self ):
        ### uses the super class __init__
        super().__init__()
        nrec = nmrec # Number of outputs that will be record
        self.SetSymbTimeUnits(d) # días
        ### Add variables ###

        ## State variables
        for name in STATE_VARS:
           STATE_VARS[name].addvar_rhs(self)

        ## Inputs
        for name in INPUTS:
            INPUTS[name].addvar_rhs(self)
        
        ## Constants
        for name in STOMAL_RESISTANCE:
           STOMAL_RESISTANCE[name].addvar_rhs(self)

        ## CO2 absorption
        for name in CO2_ABSORPTION:
           CO2_ABSORPTION[name].addvar_rhs(self)

        ## Assimilates
        for name in ASSIMILATES:
           ASSIMILATES[name].addvar_rhs(self)

    def RHS(self, Dt):
        """RHS( Dt ) = \kappa_1^{-1} F_1( t+Dt, X+k) where X is the current value of
           all state variables.  k is a simple dictionary { 'v1':k1, 'v2':k2 ... etc}
           
           ************* JUST CALL STATE VARIABLES WITH self.Vk ******************
           
           Use from ModMod TranslateArgNames() for guide you how call the functions 
        """
        ## Cálculos de la resitencia estomática
        f_R1 = f_R( I=self.V('PAR'), C_ev1=self.V('C_ev1'), C_ev2=self.V('C_ev2') )
        Sr1 = Sr( I=self.V('PAR'), S=self.V('S'), Rs=self.V('Rs') )
        C_ev31 = C_ev3( C_ev3n=self.V('C_ev3n'), C_ev3d=self.V('C_ev3d'), Sr=Sr1 )
        f_C1 = f_C( C_ev3=C_ev31, C1=self.V('C1'), k_fc=self.V('k_fc') ) 
        C_ev41 = C_ev4( C_ev4n=self.V('C_ev4n'), C_ev4d=self.V('C_ev4d'), Sr=Sr1 )
        V_sa1 = V_sa( T =self.V('T') ) # V1 -> nos interesa
        VPD1 = VPD( V_sa=V_sa1, RH=self.V('RH') )
        #breakpoint()
        f_V1 = f_V( C_ev4=C_ev41, VPD = VPD1)
        R_s1 = r_s( r_m=self.V('r_m'), f_R=f_R1, f_C=f_C1, f_V=f_V1, k_d=self.V('k_d') ) 
        ## Cálculos absorción de CO2
        g_s = gTC( k=self.V('ks'), Rb=self.V('Rb'), Rs=R_s1, k_d=self.V('k_d') )
        Ca1 = Ca( gtc=g_s, C=self.V('C1'), Ci=self.Vk('Ci') )
        Dt_Ci = ( Ca1 - (1e-3)*self.V('A') )/0.554 # Los asimilados se pasan a mg/m**2 y el incremento del Ci queda en ppm
        return Dt_Ci
