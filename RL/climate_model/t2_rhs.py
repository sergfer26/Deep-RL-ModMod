from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, BETA, GAMMA, EPSIL, ETA, LAMB, RHO, TAU, NU, PHI, OMEGA
from .constants import CONSTANTS, INPUTS, INIT_STATE, nmrec, theta
from .functions import b1
from .functions import q6
from .functions import l2 
from .functions import g3, g4
from .functions import kappa2
from .functions import n1, n2, n3
from .functions import r8, r9, r10, r11, r12, r13
from .functions import f1, f2, f3, f4, f5, f6, f7
from .functions import h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11


sigma = CONSTANTS['sigma']
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()
C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, m_cover, kg_air = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm m_cover kg_air')  

########### T2 ############
class T2_rhs(StateRHS):
    """Define a RHS, this is the rhs for T2, Greenhouse air temperature"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U7', prn=r'$U_7$',
                    desc="Forced ventilation control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U8', prn=r'$U_8$',
                    desc="Roof vents control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U3', prn=r'$U_3$',
                    desc="Control of mechanical cooling system", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U4', prn=r'$U_4$',
                    desc="Air heater control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U9', prn=r'$U_9$',
                    desc="Fog system control", units=1, val=0)
        # Inputs
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val=I6)
        self.AddVar(typ='State', varid='I2', prn=r'$I_2$',
                    desc="External global radiation", units=W * m**-2, val=I2)
        self.AddVar(typ='State', varid='I3', prn=r'$I_3$',
                    desc="Heating pipe temperature", units=C, val=I3)
        self.AddVar(typ='State', varid='I4', prn=r'$I_4$',
                    desc="Sky temperature", units=C, val=I4)
        self.AddVar(typ='Cnts', varid='I7', prn=r'$I_7$',
                    desc="Soil temperature", units=C, val=I7) # Valor tomado de internet
        # Constants
        self.AddVar(typ='Cnts', varid='tau3', prn=r'$\tau_3$',
                    desc="FIR transmission coefficient of the thermal screen", units=1, val=TAU['tau3']) # ok --> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=PHI['phi7']) # ok
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=ALPHA['alpha6']) # ok
        self.AddVar(typ='Cnts', varid='eta6', prn=r'$\eta_6$',
                    desc="Ventilation power reduction factor", units=m**3 * m**-2 * s**-1, val=ETA['eta6']) # Falta valor
        self.AddVar(typ='Cnts', varid='eta7', prn=r'$\eta_7$',
                    desc="Ratio between ceiling ventilation area and total ventilation area", units=1, val=ETA['eta7']) # no dan valor en el artículo
        self.AddVar(typ='Cnts', varid='eta8', prn=r'$\eta_8$',
                    desc="Ratio between ceiling and total ventilation area, if there is no chimney effect", units=1, val=ETA['eta8']) # ok
        self.AddVar(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=PHI['phi8']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=NU['nu4']) # ok
        self.AddVar(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=NU['nu5']) # 0.2*alpha6 --> ok
        self.AddVar(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=OMEGA['omega1']) # ok
        self.AddVar(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=NU['nu6']) # ok
        self.AddVar(typ='Cnts', varid='beta3', prn=r'$\beta_3$',
                    desc="Canopy extinction coefficient for NIR radiation", units=1, val=BETA['beta3']) # ok
        self.AddVar(typ='Cnts', varid='gamma1', prn=r'$\gamma_1$',
                    desc="Length of the heating pipe", units=m * m**-2, val=GAMMA['gamma1']) # ok ---> Usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi1', prn=r'$\phi_1$',
                    desc="External diameter of the heating pipe", units=m, val=PHI['phi1']) # ok
        self.AddVar(typ='Cnts', varid='tau1', prn=r'$\tau_1$',
                    desc="PAR transmission coefficient of the Cover", units=1, val=TAU['tau1']) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='tau2', prn=r'$\tau_2$',
                    desc="FIR transmission coefficient of the Cover", units=1, val=TAU['tau2']) # En el artículo no dan su valor
        self.AddVar(typ='Cnts', varid='lamb5', prn=r'$\lambda_5$',
                    desc="Cover surface", units=m**2, val=LAMB['lamb5']) # ok --> tomé el valor de Holanda, el de Texas es muy grande (9e4)
        self.AddVar(typ='Cnts', varid='lamb6', prn=r'$\lambda_6$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", 
                    units=W * m_cover**-2 * K**-1, val=LAMB['lamb6']) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='lamb7', prn=r'$\lambda_7$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", 
                    units=J * m**-3 * K**-1, val=LAMB['lamb7']) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='lamb8', prn=r'$\lambda_8$',
                    desc="Variable of heat exchange by convection between the roof and the outside air", 
                    units=1, val=LAMB['lamb8']) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=ALPHA['alpha5']) # ok
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= RHO['rho3']) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=NU['nu1']) # ok
        self.AddVar(typ='Cnts', varid='eta10', prn=r'$\eta_{10}$',
                    desc="Shadow effect on the discharge coefficient", units=1, val=ETA['eta10']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=NU['nu3']) # ok, en ejemplos del artículo usan valor cero
        self.AddVar(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=NU['nu2']) # ok
        self.AddVar(typ='Cnts', varid='eta11', prn=r'$\eta_{11}$',
                    desc="Effect of shadow on the global wind pressure coefficient", units=1, val=ETA['eta11']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha8', prn=r'$\alpha_8$',
                    desc="PAR absorption coefficient of the cover", units=1, val=ALPHA['alpha8']) # En el artículo no dan el valor
        self.AddVar(typ='Cnts', varid='alpha9', prn=r'$\alpha_9$',
                    desc="NIR absorption coefficient of the cover", units=1, val=ALPHA['alpha9']) # En el artículo no dan el valor
        self.AddVar(typ='Cnts', varid='eta2', prn=r'$\eta_2$',
                    desc="Ratio between PAR radiation and external global radiation", units=1, val=ETA['eta2']) # ok
        self.AddVar(typ='Cnts', varid='eta3', prn=r'$\eta_3$',
                    desc="Ratio between NIR radiation and global external radiation", units=1, val=ETA['eta3']) # ok
        self.AddVar(typ='Cnts', varid='lamb', prn=r'$\lambda$',
                    desc="Stefan-Boltzmann constant", units=W * m**-2 * K**-4, val=sigma) # ok
        self.AddVar(typ='Cnts', varid='epsil3', prn=r'$\epsilon_3$',
                    desc="Sky FIR emission coefficient", units=1, val=EPSIL['epsil3'])  # ok
        self.AddVar(typ='Cnts', varid='epsil4', prn=r'$\epsilon_4$',
                    desc="Floor FIR emission coefficient", units=1, val=EPSIL['epsil4']) # ok
        self.AddVar(typ='Cnts', varid='epsil5', prn=r'$\epsilon_5$',
                    desc="Thermal screen FIR emission coefficient", units=1, val=EPSIL['epsil5'])
        self.AddVar(typ='Cnts', varid='epsil6', prn=r'$\epsilon_6$',
                    desc="External cover FIR emission coefficient", units=1, val=EPSIL['epsil6']) # ok ---> usé el valor de Texas
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1]) # Valor original 4
        self.AddVar(typ='Cnts', varid='alpha4', prn=r'$\alpha_4$',
                    desc="Convection heat exchange coefficient of canopy leaf to greenhouse air", 
                    units=W * m**-2 * K**-1, val=alpha4) # ok
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=GAMMA['gamma2']) # ok
        self.AddVar(typ='Cnts', varid='eta5', prn=r'$\eta_5$',
                    desc="Fan-pad system efficiency", units=1, val=ETA['eta5']) # Falta valor
        self.AddVar(typ='Cnts', varid='phi5', prn=r'$\phi_5$',
                    desc="Water vapor contained in the fan-pad system", units=kg_water * kg_air**-1, val=PHI['phi5']) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi6', prn=r'$\phi_6$',
                    desc="Water vapor contained in the outside air", units=kg_water * kg_air**-1, val=PHI['phi6']) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='lamb1', prn=r'$\lambda_1$',
                    desc="Performance coefficient of the mechanical acceleration system", units=1, val=LAMB['lamb1']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb2', prn=r'$\lambda_2$',
                    desc="Electrical capacity of the mechanical cooling system", units=W, val=LAMB['lamb2']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb3', prn=r'$\lambda_3$',
                    desc="Convictive heat exchange coefficient between soil and greenhouse air", units=W * m**-2 * K**-1, val=LAMB['lamb3']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=LAMB['lamb4']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha2', prn=r'$\alpha_2$',
                    desc="Global NIR absorption coefficient of the canopy", units=1, val=ALPHA['alpha2']) # ok
        self.AddVar(typ='Cnts', varid='alpha7', prn=r'$\alpha_7$',
                    desc="Global NIR absorption coefficient of the floor", units=1, val=ALPHA['alpha7']) # ok
        self.AddVar(typ='Cnts', varid='eta1', prn=r'$\eta_1$',
                    desc="Proportion of global radiation that is absorbed by greenhouse building elements", units=1, val=ETA['eta1']) # ok
        self.AddVar(typ='Cnts', varid='phi9', prn=r'$\phi_9$',
                    desc="Fog system capacity", units=kg_water * s**-1, val=PHI['phi9']) # Falta valor
        self.AddVar(typ='Cnts', varid='nu7', prn=r'$\nu_7$',
                    desc="Soil thermal conductivity", units=W * m**-1 * K**-1, val=NU['nu7']) # ok
        self.AddVar(typ='Cnts', varid='nu8', prn=r'$\nu_8$',
                    desc="Floor to ground distance", units=m, val=NU['nu8']) # ok
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
        b_1 = b1(U1=self.V('U1'), tau3=self.V('tau3'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        n_1 = n1(U5=self.V('U5'), nu1=self.V('nu1'), eta10=self.V('eta10'))
        n_2 = n2(U6=self.V('U6'), nu3=self.V('nu3'))
        n_3 = n3(U5=self.V('U5'), nu2=self.V('nu2'), eta11=self.V('eta11'))
        f_5 = f5(I8=self.V('I8'), alpha6=self.V(
            'alpha6'), n1=n_1, n2=n_2, n3=n_3)
        f_6 = f6(I8=self.V('I8'), nu4=self.V('nu4'))
        f_2 = f2(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f5=f_5, f6=f_6)
        f_3 = f3(U7=self.V('U7'), phi8=self.V(
            'phi8'), alpha6=self.V('alpha6'))
        f_7 = f7(T2=self.Vk('T2'), U8=self.V('U8'), I5=self.V('I5'), I8=self.V('I8'), nu5=self.V(
            'nu5'), alpha6=self.V('alpha6'), omega1=self.V('omega1'), nu6=self.V('nu6'), n1=n_1, n3=n_3)
        f_4 = f4(U1=self.V('U1'), eta6=self.V('eta6'), eta7=self.V(
            'eta7'), eta8=self.V('eta8'), f6=f_6, f7=f_7)
        g_3 = g3(I1=self.V('I1'), beta3=self.V('beta3'), gamma1=self.V(
            'gamma1'), phi1=self.V('phi1'), tau1=self.V('tau1'), b1=b_1)
        g_4 = g4(U1=self.V('U1'), tau2=self.V('tau2'))
        h_8 = h8(T2=self.Vk('T2'), I5=self.V('I5'), I8=self.V('I8'), alpha6=self.V('alpha6'), lamb5=self.V(
            'lamb5'), lamb6=self.V('lamb6'), lamb7=self.V('lamb7'), lamb8=self.V('lamb8'))
        h_9 = h9(T2=self.Vk('T2'), I5=self.V('I5'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f4=f_4)
        q_6 = q6(I6=self.V('I6'))
        r_9 = r9(I2=self.V('I2'), alpha8=self.V('alpha8'), alpha9=self.V(
            'alpha9'), eta2=self.V('eta2'), eta3=self.V('eta3'))
        r_11 = r11(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil4=self.V('epsil4'), g3=g_3)
        r_12 = r12(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil5=self.V('epsil5'), g4=g_4)
        r_13 = r13(T2=self.Vk('T2'), I4=self.V('I4'), lamb=self.V(
            'lamb'), epsil3=self.V('epsil3'), epsil6=self.V('epsil6'))
        #### Principal functions ####
        kappa_2 = kappa2(alpha5=self.V('alpha5'),
                         rho3=self.V('rho3'), phi2=self.V('phi2'))
        h_1 = h1(T1=self.Vk('T1'), T2=self.Vk('T2'),
                 I1=self.V('I1'), alpha4=self.V('alpha4'))
        h_2 = h2(I5=self.V('I5'), alpha5=self.V('alpha5'), gamma2=self.V('gamma2'), eta5=self.V(
            'eta5'), rho3=self.V('rho3'), phi5=self.V('phi5'), phi6=self.V('phi6'), f1=f_1)
        h_3 = h3(T2=self.Vk('T2'), V1=self.Vk('V1'), U3=self.V('U3'), I6=self.V('I6'), lamb1=self.V(
            'lamb1'), lamb2=self.V('lamb2'), alpha6=self.V('alpha6'), gamma2=self.V('gamma2'), q6=q_6)
        h_4 = h4(T2=self.Vk('T2'), I3=self.V('I3'),
                 gamma1=self.V('gamma1'), phi1=self.V('phi1'))
        h_5 = h5(T2=self.Vk('T2'), I7=self.V('I7'), lamb3=self.V('lamb3'))
        h_6 = h6(U4=self.V('U4'), lamb4=self.V(
            'lamb4'), alpha6=self.V('alpha6'))
        r_8 = r8(I2=self.V('I2'), alpha2=self.V('alpha2'), alpha7=self.V('alpha7'), eta1=self.V(
            'eta1'), eta2=self.V('eta2'), eta3=self.V('eta3'), tau1=self.V('tau1'), r9=r_9)
        h_7 = h7(T2=self.Vk('T2'), I5=self.V('I5'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f2=f_2, f3=f_3, h8=h_8, h9=h_9)
        h_10 = h10(T2=self.Vk('T2'), alpha5=self.V(
            'alpha5'), rho3=self.V('rho3'), f1=f_1)
        l_2 = l2(U9=self.V('U9'), alpha6=self.V('alpha6'),
                 gamma2=self.V('gamma2'), phi9=self.V('phi9'))
        r_10 = r10(r11=r_11, r12=r_12, r13=r_13)
        h_11 = h11(T2=self.Vk('T2'), I7=self.V('I7'), nu7=self.V(
            'nu7'), nu8=self.V('nu8'), phi2=self.V('phi2'))
        return (kappa_2**-1)*(h_1 + h_2 + h_3 + h_4 + h_5 + h_6 + r_8 - h_7 - h_10 - l_2 - r_10 - h_11)