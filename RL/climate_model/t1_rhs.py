from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, BETA, GAMMA, ETA, TAU, RHO, DELTA, EPSIL
from .constants import CONSTANTS, INPUTS, INIT_STATE, nmrec, theta
from .functions import a1
from .functions import b1
from .functions import p1
from .functions import l1
from .functions import h1
from .functions import g1, g2
from .functions import kappa1
from .functions import r1, r2, r3, r4, r5, r6, r7 
from .functions import q1, q2, q3, q4, q5, q7, q8, q9, q10


# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm')

################## Constants ##################
sigma = CONSTANTS['sigma']
tau1, tau2, tau3 = TAU.values()
rho1, rho2, rho3, _ = RHO.values()
beta1, beta2, beta3 = BETA.values()
_, _, _, _, _, delta6, delta7 = DELTA.values()
epsil1, epsil2, epsil3, _, _, _ = EPSIL.values()
gamma, _, gamma2, gamma3, gamma4, gamma5 = GAMMA.values()
_, alpha2, alpha3, alpha4, alpha5, _, _, _ = ALPHA.values()
delta1, delta2, delta3, delta4, delta5, _, _ = DELTA.values()
eta1, eta2, eta3, eta4, _, _, _, _, _, _, _, _, _ = ETA.values()

################## Inputs ##################
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()

################## State variables ##################
C1_in, V1_in, T1_in, T2_in = INIT_STATE.values()


class T1_rhs(StateRHS):
    """Define a RHS, this is the rhs for C1, the CO2 concentrartion in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec) # falta valor inicial
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=0)
        # Inputs
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I2', prn=r'$I_2$',
                    desc="External global radiation", units=W * m**-2, val=I2)
        self.AddVar(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=I3)
        self.AddVar(typ='State', varid='I4', prn=r'$I_4$',
                    desc="Sky temperature", units=C, val=I4)
        self.AddVar(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=I14)
        # Constants
        self.AddVar(typ='Cnts', varid='beta3', prn=r'$\beta_3$',
                    desc="Canopy extinction coefficient for NIR radiation", units=1, val=beta3) # ok
        self.AddVar(typ='Cnts', varid='tau3', prn=r'$\tau_3$',
                    desc="FIR transmission coefficient of the thermal screen", units=1, val=tau3) # ok --> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='beta1', prn=r'$\beta_1$',
                    desc="Canopy extinction coefficient for PAR radiation", units=1, val=beta1) # ok
        self.AddVar(typ='Cnts', varid='rho1', prn=r'$\rho_1$',
                    desc="PAR reflection coefficient", units=1, val=rho1) # ok
        self.AddVar(typ='Cnts', varid='beta2', prn=r'$\beta_2$',
                    desc="Extinction coefficient for PAR radiation reflected from the floor to the canopy", units=1, val=beta2) # ok
        self.AddVar(typ='Cnts', varid='rho2', prn=r'$\rho_2$',
                    desc="Floor reflection coefficient PAR", units=1, val=rho2) # ok
        self.AddVar(typ='Cnts', varid='eta1', prn=r'$\eta_1$',
                    desc="Proportion of global radiation that is absorbed by greenhouse building elements", units=1, val=eta1) # ok
        self.AddVar(typ='Cnts', varid='eta2', prn=r'$\eta_2$',
                    desc="Ratio between PAR radiation and external global radiation", units=1, val=eta2) # ok
        self.AddVar(typ='Cnts', varid='tau1', prn=r'$\tau_1$',
                    desc="PAR transmission coefficient of the Cover", units=1, val=tau1) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='tau2', prn=r'$\tau_2$',
                    desc="FIR transmission coefficient of the Cover", units=1, val=tau2) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= rho3) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=alpha5) # ok
        self.AddVar(typ='Cnts', varid='gamma', prn=r'$\gamma$',
                    desc="Psychometric constan", units=Pa * K**-1, val=gamma) # ok
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=gamma2) # ok
        self.AddVar(typ='Cnts', varid='gamma3', prn=r'$\gamma_3$',
                    desc="Strength of boundary layer of canopy for vapor transport", units=s * m**-1, val=gamma3) # ok
        self.AddVar(typ='Cnts', varid='delta1', prn=r'$\delta_1$',
                    desc="Radiation above the canopy that defines sunrise and sunset", units=W * m**-2, val=delta1) # ok
        self.AddVar(typ='Cnts', varid='delta2', prn=r'$\delta_2$',
                    desc="Empirically determined parameter", units=W * m**-2, val=delta2) # ok
        self.AddVar(typ='Cnts', varid='delta3', prn=r'$\delta_3$',
                    desc="Empirically determined parameter", units=W * m**-2, val=delta3) # ok
        self.AddVar(typ='Cnts', varid='gamma4', prn=r'$\gamma_4$',
                    desc="Minimum stomatal resistance of the canopy", units=s * m**-1, val=gamma4) # ok
        self.AddVar(typ='Cnts', varid='gamma5', prn=r'$\gamma_5$',
                    desc="Slope of the differentiable switch for the stomatal resistance model", units=m * W**-2, val=gamma5) # ok
        self.AddVar(typ='Cnts', varid='delta4', prn=r'$\delta_4$',
                    desc="Coefficient of the CO2 transpiration in the day", units=ppm**-2, val=delta4) # ok
        self.AddVar(typ='Cnts', varid='delta5', prn=r'$\delta_5$',
                    desc="Coefficient of the CO2 transpiration in the night", units=ppm**-2, val=delta5) # ok
        self.AddVar(typ='Cnts', varid='delta6', prn=r'$\delta_6$',
                    desc="Coefficient of the vapour pressure in the day", units=Pa**-2, val=delta6) # ok
        self.AddVar(typ='Cnts', varid='delta7', prn=r'$\delta_7$',
                    desc="Coefficient of the vapour pressure in the night", units=Pa**-2, val=delta7) # ok
        self.AddVar(typ='Cnts', varid='eta4', prn=r'$\eta_4$',
                    desc="Conversion factor for CO2 of mg*m**−3 to ppm", units=ppm * mg**-1 * m**3, val=eta4) # ok
        self.AddVar(typ='Cnts', varid='alpha1', prn=r'$\alpha_1$',
                    desc="Heat capacity of one square meter of the canopy", units=J * K**-1 * m**-2, val=theta[0]) # Valor original 1.2e3
        self.AddVar(typ='Cnts', varid='alpha2', prn=r'$\alpha_2$',
                    desc="Global NIR absorption coefficient of the canopy", units=1, val=alpha2) # ok
        self.AddVar(typ='Cnts', varid='eta3', prn=r'$\eta_3$',
                    desc="Ratio between NIR radiation and global external radiation", units=1, val=eta3) # ok
        self.AddVar(typ='Cnts', varid='alpha3', prn=r'$\alpha_3$',
                    desc="Surface of the heating pipe", units=m**2*m**-2, val=alpha3) # Valor proporcionado por Dr Antonio
        self.AddVar(typ='Cnts', varid='epsil1', prn=r'$\epsilon_1$',
                    desc="FIR emission coefficient of the heating pipe", units=1, val=epsil1) # ok
        self.AddVar(typ='Cnts', varid='epsil2', prn=r'$\epsilon_2$',
                    desc="Canopy FIR emission coefficient", units=1, val=epsil2) # ok
        self.AddVar(typ='Cnts', varid='lamb', prn=r'$\sigma$',
                    desc="Stefan-Boltzmann constant", units=W * m**-2 * K**-4, val=sigma) # ok
        self.AddVar(typ='Cnts', varid='alpha4', prn=r'$\alpha_4$',
                    desc="Convection heat exchange coefficient of canopy leaf to greenhouse air", units=W * m**-2 * K**-1, val=alpha4) # ok
        self.AddVar(typ='Cnts', varid='epsil3', prn=r'$\epsilon_3$',
                    desc="Sky FIR emission coefficient", units=1, val=epsil3)  # ok
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
        a_1 = a1(I1=self.V('I1'), beta3=self.V('beta3'))
        b_1 = b1(U1=self.V('U1'), tau3=self.V('tau3'))
        r_4 = r4(I2=self.V('I2'), eta1=self.V('eta1'),
                 eta2=self.V('eta2'), tau1=self.V('tau1'))
        r_2 = r2(I1=self.V('I1'), beta1=self.V(
            'beta1'), rho1=self.V('rho1'), r4=r_4)
        r_3 = r3(I1=self.V('I1'), beta1=self.V('beta1'), beta2=self.V(
            'beta2'), rho1=self.V('rho1'), rho2=self.V('rho2'), r4=r_4)
        g_1 = g1(a1=a_1)
        g_2 = g2(tau2=self.V('tau2'), b1=b_1)
        q_2 = q2(T1=self.Vk('T1'))
        q_7 = q7(I14=self.V('I14'), delta1=self.V(
            'delta1'), gamma5=self.V('gamma5'))
        q_8 = q8(delta4=self.V('delta4'), delta5=self.V('delta5'), q7=q_7)
        q_9 = q9(delta6=self.V('delta6'), delta7=self.V('delta7'), q7=q_7)
        q_4 = q4(C1=self.V('C1'), eta4=self.V('eta4'), q8=q_8)
        q_5 = q5(V1=self.V('V1'), q2=q_2, q9=q_9)
        q_10 = q10(I14=self.V('I14'), delta2=self.V(
            'delta2'), delta3=self.V('delta3'))
        q_3 = q3(I14=self.V('I14'), gamma4=self.V(
            'gamma4'), q4=q_4, q5=q_5, q10=q_10)
        q_1 = q1(I1=self.V('I1'), rho3=self.V('rho3'), alpha5=self.V('alpha5'), gamma=self.V(
            'gamma'), gamma2=self.V('gamma2'), gamma3=self.V('gamma3'), q3=q_3)
        p_1 = p1(V1=self.Vk('V1'), q1=q_1, q2=q_2)
        #### Principal functions ####
        kappa_1 = kappa1(I1=self.V('I1'), alpha1=self.V('alpha1'))
        r_1 = r1(r2=r_2, r3=r_3)
        r_5 = r5(I2=self.V('I2'), alpha2=self.V('alpha2'),
                 eta1=self.V('eta1'), eta3=self.V('eta3'))
        r_6 = r6(T1=self.Vk('T1'), I3=self.V('I3'), alpha3=self.V('alpha3'), epsil1=self.V(
            'epsil1'), epsil2=self.V('epsil2'), lamb=self.V('lamb'), g1=g_1)
        h_1 = h1(T1=self.Vk('T1'), T2=self.Vk('T2'),
                 I1=self.V('I1'), alpha4=self.V('alpha4'))
        l_1 = l1(gamma2=self.V('gamma2'), p1=p_1)
        r_7 = r7(T1=self.Vk('T1'), I4=self.V('I4'), epsil2=self.V(
            'epsil2'), epsil3=self.V('epsil3'), lamb=self.V('lamb'), a1=a_1, g2=g_2)
        return (kappa_1**-1)*(r_1 + r_5 + r_6 - h_1 - l_1 - r_7)