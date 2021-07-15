from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, GAMMA, DELTA, ETA, LAMB, RHO, NU, PHI, PSI, OMEGA
from .constants import INPUTS, INIT_STATE, nmrec, theta
from .functions import h3, h6
from .functions import kappa3
from .functions import f1, f2, f3, f4, f5, f6, f7
from .functions import q1, q2, q3, q4, q5, q6, q7, q8, q9, q10
from .functions import p1, p2, p3, p4, p5, p6, p7
from .functions import n1, n2, n3


I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()
C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, m_cover, kg_air = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm m_cover kg_air') 


class V1_rhs(StateRHS):
    """Define a RHS, this is the rhs for V1, the vapour pression in the greenhouse air"""
    def __init__(self):

        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec) # Falta valor inical
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec) # falta valor inicial
        # control variables ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U4', prn=r'$U_4$',
                    desc="Air heater control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U1', prn=r'$U_1$',
                    desc="Thermal screen control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U7', prn=r'$U_7$',
                    desc="Forced ventilation control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U8', prn=r'$U_8$',
                    desc="Roof vents control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U3', prn=r'$U_3$',
                    desc="Control of mechanical cooling system", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U9', prn=r'$U_9$',
                    desc="Fog system control", units=1, val=0)
        # Inputs
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val=I6)
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1) # Valor tomado de internet
        self.AddVar(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=I14)
        # Constants
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=LAMB['lamb4']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=ALPHA['alpha6']) # ok
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=PHI['phi7']) # ok
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
        self.AddVar(typ='Cnts', varid='lamb1', prn=r'$\lambda_1$',
                    desc="Performance coefficient of the mechanical acceleration system", units=1, val=LAMB['lamb1']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb2', prn=r'$\lambda_2$',
                    desc="Electrical capacity of the mechanical cooling system", units=W, val=LAMB['lamb2']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=GAMMA['gamma2']) # ok
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
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val= RHO['rho3']) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=ALPHA['alpha5']) # ok
        self.AddVar(typ='Cnts', varid='gamma', prn=r'$\gamma$',
                    desc="Psychometric constan", units=Pa * K**-1, val=GAMMA['gamma']) # ok
        self.AddVar(typ='Cnts', varid='gamma3', prn=r'$\gamma_3$',
                    desc="Strength of boundary layer of canopy for vapor transport", units=s * m**-1, val=GAMMA['gamma3']) # ok
        self.AddVar(typ='Cnts', varid='delta1', prn=r'$\delta_1$',
                    desc="Radiation above the canopy that defines sunrise and sunset", units=W * m**-2, val=DELTA['delta1']) # ok
        self.AddVar(typ='Cnts', varid='delta2', prn=r'$\delta_2$',
                    desc="Empirically determined parameter", units=W * m**-2, val=DELTA['delta2']) # ok
        self.AddVar(typ='Cnts', varid='delta3', prn=r'$\delta_3$',
                    desc="Empirically determined parameter", units=W * m**-2, val=DELTA['delta3']) # ok
        self.AddVar(typ='Cnts', varid='gamma4', prn=r'$\gamma_4$',
                    desc="Minimum stomatal resistance of the canopy", units=s * m**-1, val=GAMMA['gamma4']) # ok
        self.AddVar(typ='Cnts', varid='gamma5', prn=r'$\gamma_5$',
                    desc="Slope of the differentiable switch for the stomatal resistance model", units=m * W**-2, val=GAMMA['gamma5']) # ok
        self.AddVar(typ='Cnts', varid='delta4', prn=r'$\delta_5$',
                    desc="Coefficient of the CO2 transpiration in the day", units=ppm**-2, val=DELTA['delta5']) # ok
        self.AddVar(typ='Cnts', varid='delta5', prn=r'$\delta_6$',
                    desc="Coefficient of the CO2 transpiration in the night", units=ppm**-2, val=DELTA['delta6']) # ok
        self.AddVar(typ='Cnts', varid='delta6', prn=r'$\delta_7$',
                    desc="Coefficient of the vapour pressure in the day", units=Pa**-2, val=DELTA['delta7']) # ok
        self.AddVar(typ='Cnts', varid='delta7', prn=r'$\delta_8$',
                    desc="Coefficient of the vapour pressure in the night", units=Pa**-2, val=DELTA['delta8']) # ok
        self.AddVar(typ='Cnts', varid='eta4', prn=r'$\eta_4$',
                    desc="Conversion factor for CO2 of mg*m**−3 to ppm", units=ppm * mg**-1 * m**3, val=ETA['eta4']) # ok
        self.AddVar(typ='Cnts', varid='psi1', prn=r'$\psi_1$',
                    desc="Molar mass of water", units=kg * kmol**-1, val=PSI['psi1']) # ok
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1]) # Valor original 4
        self.AddVar(typ='Cnts', varid='omega2', prn=r'$\omega_2$',
                    desc="Molar gas constant", units=J * kmol**-1 * K**-1, val=OMEGA['omega2']) # ok
        self.AddVar(typ='Cnts', varid='eta5', prn=r'$\eta_5$',
                    desc="Fan-pad system efficiency", units=1, val=ETA['eta5']) # Falta valor
        self.AddVar(typ='Cnts', varid='phi5', prn=r'$\phi_5$',
                    desc="Water vapor contained in the fan-pad system", units=kg_water * kg_air**-1, val=PHI['phi5']) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi6', prn=r'$\phi_6$',
                    desc="Water vapor contained in the outside air", units=kg_water * kg_air**-1, val=PHI['phi6']) # Falta valor --> En realaidad es un input
        self.AddVar(typ='Cnts', varid='phi9', prn=r'$\phi_9$',
                    desc="Fog system capacity", units=kg_water * s**-1, val=PHI['phi9']) # Falta valor
        self.AddVar(typ='Cnts', varid='eta12', prn=r'$\eta_{12}$',
                    desc="Amount of vapor that is released when a joule of sensible energy is produced by the direct air heater", 
                    units=kg_vapour * J**-1, val=ETA['eta12']) # ok
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
        h_6 = h6(U4=self.V('U4'), lamb4=self.V(
            'lamb4'), alpha6=self.V('alpha6'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V(
            'phi7'), alpha6=self.V('alpha6'))
        f_3 = f3(U7=self.V('U7'), phi8=self.V(
            'phi8'), alpha6=self.V('alpha6'))
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
        p_4 = p4(eta12=self.V('eta12'), h6=h_6)
        p_5 = p5(T2=self.Vk('T2'), V1=self.Vk('V1'), I5=self.V('I5'), psi1=self.V(
            'psi1'), omega2=self.V('omega2'), f2=f_2, f3=f_3, f4=f_4)
        p_6 = p6(T2=self.Vk('T2'), V1=self.Vk('V1'), psi1=self.V(
            'psi1'), omega2=self.V('omega2'), f1=f_1)
        p_7 = p7(V1=self.Vk('V1'), h3=h_3, q6=q_6)
        return (kappa_3**-1)*(p_1 + p_2 + p_3 + p_4 - p_5 - p_6 - p_7)