from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, GAMMA, DELTA, ETA, LAMB, RHO, PHI, PSI, OMEGA
from .constants import CONSTANTS, INPUTS, INIT_STATE, nmrec, theta
from .functions import f1
from .functions import h3
from .functions import kappa3
from .functions import p1, p2, p3
from .functions import q1, q2, q3, q4, q5, q6, q7, q8, q9, q10


etadrain = CONSTANTS['etadrain']
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()
C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, mxn= symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm mxn')


class Qh2o_rhs(StateRHS):
    """Define a RHS, this is the rhs for Qh20, the water cost per kg"""
    def __init__(self):
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='Qh20', prn=r'$Q_{H20}$',
                    desc="Water cost ", units=mxn * kg, val=0, rec=nrec)
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='T1', prn=r'$T_1$',
                    desc="Canopy temperature", units=C, val=T1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec) # falta valor inicial
        self.AddVar(typ='State', varid='V1', prn=r'$V_1$',
                    desc="Greenhouse air vapor pressure", units=Pa, val=V1_in, rec=nrec)
        # control variables 
        self.AddVar(typ='Cnts', varid='U2', prn=r'$U_2$',
                    desc="Fan and pad system control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U3', prn=r'$U_3$',
                    desc="Control of mechanical cooling system", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U9', prn=r'$U_9$',
                    desc="Fog system control", units=1, val=0)
        # Inputs
        self.AddVar(typ='Cnts', varid='I1', prn=r'$I_1$',
                    desc="Leaf area index", units=m**2 * m**-2, val=I1)
        self.AddVar(typ='State', varid='I6', prn=r'$I_6$',
                    desc="Mechanical cooling system temperature", units=C, val=I6)
        
        self.AddVar(typ='State', varid='I14', prn=r'$\I_{14}$',
                    desc="Global radiation above the canopy", units=W * m**-2, val=I14)
        # Constants
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=ALPHA['alpha6']) # ok
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=PHI['phi7']) # ok
        self.AddVar(typ='Cnts', varid='lamb1', prn=r'$\lambda_1$',
                    desc="Performance coefficient of the mechanical acceleration system", units=1, val=LAMB['lamb1']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='lamb2', prn=r'$\lambda_2$',
                    desc="Electrical capacity of the mechanical cooling system", units=W, val=LAMB['lamb2']) # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='gamma2', prn=r'$\gamma_2$',
                    desc="Latent heat of water evaporation", units=J * kg_water**-1, val=GAMMA['gamma2']) # ok
        self.AddVar(typ='Cnts', varid='rho3', prn=r'$\rho_3$',
                    desc="Air density", units=kg * m**-3, val=RHO['rho3']) # El valor es el de la densidad del aire al nivel del mar
        self.AddVar(typ='Cnts', varid='alpha5', prn=r'$\alpha_5$',
                    desc="Specific heat capacity of greenhouse air", units=J * K**-1 * kg**-1, val=ALPHA['alpha5']) # ok
        self.AddVar(typ='Cnts', varid='gamma', prn=r'$\gamma$',
                    desc="Psychometric constan", units=Pa * K**-1, val=GAMMA['gamma']) # ok
        self.AddVar(typ='Cnts', varid='gamma3', prn=r'$\gamma_3$',
                    desc="Strength of boundary layer of canopy for vapor transport", units=s * m**-1, val=GAMMA['gamma3']) # ok
        self.AddVar(typ='Cnts', varid='delta1', prn=r'$\delta_2$',
                    desc="Radiation above the canopy that defines sunrise and sunset", units=W * m**-2, val=DELTA['delta2']) # ok
        self.AddVar(typ='Cnts', varid='delta2', prn=r'$\delta_3$',
                    desc="Empirically determined parameter", units=W * m**-2, val=DELTA['delta3']) # ok
        self.AddVar(typ='Cnts', varid='delta3', prn=r'$\delta_4$',
                    desc="Empirically determined parameter", units=W * m**-2, val=DELTA['delta4']) # ok
        self.AddVar(typ='Cnts', varid='gamma4', prn=r'$\gamma_4$',
                    desc="Minimum stomatal resistance of the canopy", units=s * m**-1, val=GAMMA['gamma4']) # ok
        self.AddVar(typ='Cnts', varid='gamma5', prn=r'$\gamma_5$',
                    desc="Slope of the differentiable switch for the stomatal resistance model", units=m * W**-2, val=GAMMA['gamma5']) # ok
        self.AddVar(typ='Cnts', varid='delta4', prn=r'$\delta_4$',
                    desc="Coefficient of the CO2 transpiration in the day", units=ppm**-2, val=DELTA['delta4']) # ok
        self.AddVar(typ='Cnts', varid='delta5', prn=r'$\delta_5$',
                    desc="Coefficient of the CO2 transpiration in the night", units=ppm**-2, val=DELTA['delta6']) # ok
        self.AddVar(typ='Cnts', varid='delta6', prn=r'$\delta_6$',
                    desc="Coefficient of the vapour pressure in the day", units=Pa**-2, val=DELTA['delta6']) # ok
        self.AddVar(typ='Cnts', varid='delta7', prn=r'$\delta_7$',
                    desc="Coefficient of the vapour pressure in the night", units=Pa**-2, val=DELTA['delta7']) # ok
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
        self.AddVar(typ='Cnts', varid='etadrain', prn=r'$\eta_{drain}$',
                    desc="Missing", units=1, val=etadrain) # Falta descripción y unidades
        
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
        return (10**-3)*((1+ etadrain/100.0)*p_1 + p_2 + p_3)