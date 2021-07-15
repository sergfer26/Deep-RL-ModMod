from ModMod import StateRHS
from sympy import symbols
from .constants import ALPHA, OMEGA, LAMB, PHI, PSI, ETA, NU
from .constants import INPUTS, INIT_STATE, nmrec, theta
from .functions import f1, f2, f3, f4, f5, f6, f7
from .functions import o1, o2, o3, o4, o5, o6 
from .functions import h6, n1, n2, n3
from .functions import kappa4


# Symbolic use of base phisical units
mt, mg, m, C, s, W, mg_CO2, J, g, mol_CH2O = symbols('mt mg m C s W mg_CO2 J g mol_CH2O')

################## Constants ##################
lamb4 = LAMB['lamb4']
alpha6 = ALPHA['alpha6']
_, psi2, psi3 = PSI.values()
omega1, _, omega3 = OMEGA.values()
_, phi2, _, _, _, _, phi7, phi8, _ = PHI.values() 
nu1, nu2, nu3, nu4, nu5, nu6, _, _ = NU.values()
_, _, _, _, _, eta6, eta7, eta8, _, eta10, eta11, _, eta13 = ETA.values()

################## Inputs ##################
I1, I2, I3, I4, I5, I6, I7, I8, I10, I11, I12, I13, I14 = INPUTS.values()

################## State variables ##################
C1_in, _, _, T2_in = INIT_STATE.values()


class C1_rhs(StateRHS):
    """Define a RHS, this is the rhs for C1, the CO2 concentrartion in the greenhouse air"""
    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        # uses the super class __init__
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        nrec = nmrec  # Number of outputs that will be record
        ### Add variables ###
        # State variables
        self.AddVar(typ='State', varid='C1', prn=r'$C_1$',
                    desc="CO2 concentrartion in the greenhouse air", units=mg * m**-3, val=C1_in, rec=nrec)  # falta valor inicial
        self.AddVar(typ='State', varid='T2', prn=r'$T_2$',
                    desc="Greenhouse air temperature", units=C, val=T2_in, rec=nrec)  # falta valor inicial
        # control variables  ---> Parece que estas variables no tienen unidades, falta valores iniciales
        self.AddVar(typ='Cnts', varid='U10', prn=r'$U_{10}$',
                    desc="Control of external CO2 source", units=1, val=0)
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
        self.AddVar(typ='Cnts', varid='U5', prn=r'$U_5$',
                    desc="External shading control", units=1, val=0)
        self.AddVar(typ='Cnts', varid='U6', prn=r'$U_6$',
                    desc="Side vents Control", units=1, val=0)
        # Inputs ---> No son constantes sino variables
        self.AddVar(typ='State', varid='I8', prn=r'$I_8$',
                    desc="Outdoor wind speed", units=m * s**-1, val=I8)
        self.AddVar(typ='State', varid='I5', prn=r'$I_5$',
                    desc="Outdoor temperature", units=C, val=I5)
        self.AddVar(typ='Cnts', varid='I10', prn=r'$I_{10}$',
                    desc="Outdoor CO2 concentration", units=mg * m**-3, val=I10)
        self.AddVar(typ='Cnts', varid='I11', prn=r'$I_{11}$',
                    desc="Inhibition of the rate of photosynthesis by saturation of the leaves with carbohydrates", units=1, val=I11)  # Falta valor y unidades
        self.AddVar(typ='Cnts', varid='I12', prn=r'$I_{12}$',
                    desc="Crude canopy photosynthesis rate", units=1, val=I12)  # Falta valor y unidades
        self.AddVar(typ='Cnts', varid='I13', prn=r'$I_{13}$',
                    desc="Photorespiration during photosynthesis", units=1, val=I13)  # Falta valor y unidades
        # Constants
        self.AddVar(typ='Cnts', varid='lamb4', prn=r'$\lambda_4$',
                    desc="Heat capacity of direct air heater", units=W, val=lamb4)  # Dr Antonio dio el valor 
        self.AddVar(typ='Cnts', varid='alpha6', prn=r'$\alpha_6$',
                    desc="Greenhouse floor surface area", units=m**2, val=alpha6)  # ok
        self.AddVar(typ='Cnts', varid='phi7', prn=r'$\phi_7$',
                    desc="Capacity of air flow through the pad", units=m**3 * s**-1, val=phi7)  # ok
        self.AddVar(typ='Cnts', varid='eta6', prn=r'$\eta_6$',
                    desc="Ventilation power reduction factor", units=m**3 * m**-2 * s**-1, val=eta6)  # Falta valor
        self.AddVar(typ='Cnts', varid='eta7', prn=r'$\eta_7$',
                    desc="Ratio between ceiling ventilation area and total ventilation area", units=1, val=eta7)  # no dan valor en el artículo
        self.AddVar(typ='Cnts', varid='eta8', prn=r'$\eta_8$',
                    desc="Ratio between ceiling and total ventilation area, if there is no chimney effect", units=1, val=eta8)  # ok
        self.AddVar(typ='Cnts', varid='phi8', prn=r'$\phi_8$',
                    desc="Air flow capacity of forced ventilation system", units=m**3 * s**-1, val=phi8)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu4', prn=r'$\nu_4$',
                    desc="Leakage coefficien", units=1, val=nu4)  # ok
        self.AddVar(typ='Cnts', varid='nu5', prn=r'$\nu_5$',
                    desc="Maximum ceiling ventilation area", units=m**2, val=nu5)  # 0.2*alpha6 --> ok
        self.AddVar(typ='Cnts', varid='omega1', prn=r'$\omega_1$',
                    desc="Gravity acceleration constant", units=m * s**-2, val=omega1)  # ok
        self.AddVar(typ='Cnts', varid='nu6', prn=r'$\nu_6$',
                    desc="Vertical dimension of a single open respirator", units=m, val=nu6)  # ok
        self.AddVar(typ='Cnts', varid='nu1', prn=r'$\nu_1$',
                    desc="Shadowless discharge coefficient", units=1, val=nu1)  # ok
        self.AddVar(typ='Cnts', varid='eta10', prn=r'$\eta_{10}$',
                    desc="Shadow effect on the discharge coefficient", units=1, val=eta10)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='nu3', prn=r'$\nu_3$',
                    desc="Side surface of the greenhouse", units=m**2, val=nu3)  # ok, en ejemplos del artículo usan valor cero
        self.AddVar(typ='Cnts', varid='nu2', prn=r'$\nu_2$',
                    desc="Global wind pressure coefficient without shadow", units=1, val=nu2)  # ok
        self.AddVar(typ='Cnts', varid='eta11', prn=r'$\eta_{11}$',
                    desc="Effect of shadow on the global wind pressure coefficient", units=1, val=eta11)  # Falta valor, aunque en los ejemplos del artículo no se considera
        self.AddVar(typ='Cnts', varid='phi2', prn=r'$\phi_2$',
                    desc="Average height of greenhouse air", units=m, val=theta[1])  # Valor original 4
        self.AddVar(typ='Cnts', varid='eta13', prn=r'$\eta_{13}$',
                    desc="Amount of CO2 that is released when a joule of sensible energy is produced by the direct air heater", units=mg_CO2 * J**-1, val=eta13)  # ok
        self.AddVar(typ='Cnts', varid='psi2', prn=r'$\psi_2$',
                    desc="Capacity of the external CO2 source", units=mg * s**-1, val=theta[2])  # Falta valor, tomé el del ejemplo de Texas 4.3e5
        self.AddVar(typ='Cnts', varid='psi3', prn=r'$\psi_3$',
                    desc="Molar mass of the CH2O", units=g * mol_CH2O**-1, val=psi3)  # ok
        self.AddVar( typ='Cnts', varid='omega3', prn=r'$\omega_3$',\
                        desc="Percentage of CO2 absorbed by the canopy", units= 1 , val=omega3)
                    
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
        h_6 = h6(U4=self.V('U4'), lamb4=self.V('lamb4'), alpha6=self.V('alpha6'))
        f_1 = f1(U2=self.V('U2'), phi7=self.V('phi7'), alpha6=self.V('alpha6'))
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
        o_2 = o2(U10=self.V('U10'), psi2=self.V(
            'psi2'), alpha6=self.V('alpha6'))
        o_3 = o3(C1=self.Vk('C1'), I10=self.V('I10'), f1=f_1)
        o_4 = o4(I11=self.V('I11'), I12=self.V('I12'),
                 I13=self.V('I13'), psi3=self.V('psi3'))
        o_5 = o5(C1=self.Vk('C1'), I10=self.V(
            'I10'), f2=f_2, f3=f_3, f4=f_4)
        o_6 = o6( C1=self.Vk('C1'), omega3=self.V('omega3') )
        return (kappa_4**-1)*(o_1 + o_2 + o_3 - o_4 - o_5  - o_6)