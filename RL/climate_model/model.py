import numpy as np
from ModMod import Module, Director
from scipy.stats import norm
from .constants import INIT_STATE, CONSTANTS
from .qh2o_rhs import Qh2o_rhs
from . qco2_rhs import Qco2_rhs 
from .qgas_rhs import Qgas_rhs
from .t1_rhs import T1_rhs
from .t2_rhs import T2_rhs
from .v1_rhs import V1_rhs
from .c1_rhs import C1_rhs


C1_in, V1_in, T2_in, T1_in = INIT_STATE.values()
T_cal = CONSTANTS['T_cal']


class Module1(Module):

    def __init__(self, Dt=1, **kwargs):
        """Models one part of the process, uses the shared variables
           from Director.
           Dt=0.1, default Time steping of module
        """
        super().__init__(Dt)  # Time steping of module
        # Always, use the super class __init__, theare are several otjer initializations
        # Module specific constructors, add RHS's
        for key, value in kwargs.items():
            self.AddStateRHS(key, value)
        # print("State Variables for this module:", self.S_RHS_ids)

    def Advance(self, t1):
        # Se agrega ruido a los resultados del modelo
        s1 = 0.1131  # Desviación estándar de T1 y T2
        s2 = 0.1281  # Desviación estándar de V1
        s3 = 10  # Desviación estándar de C1
        # seed( int( self.t() ) ) # La semilla de los aleatorios depende del tiempo del director
        T1r = self.V('T1') + norm.rvs(scale=s1)
        T2r = self.V('T2') + norm.rvs(scale=s1)
        V1r = self.V('V1') + norm.rvs(scale=s2)
        C1r = self.V('C1') + norm.rvs(scale=s3)
        # Actualización de las variables
        self.V_Set('T1', T1r)
        self.V_Set('T2', T2r)
        self.V_Set('V1', V1r)
        self.V_Set('C1', C1r)
        # Avance del RHS
        self.AdvanceRungeKutta(t1)
        return 1
        

class Climate_model(Director):
    def __init__(self):
        super().__init__(t0=0.0, time_unit="", Vars={}, Modules={})
        C1_rhs_ins   = C1_rhs()     # Make an instance of rhs
        V1_rhs_ins   = V1_rhs()     # Make an instance of rhs
        T1_rhs_ins   = T1_rhs()     # Make an instance of rhs
        T2_rhs_ins   = T2_rhs()     # Make an instance of rhs
        Qgas_rhs_ins = Qgas_rhs() # Make an instance of rhs
        Qco2_rhs_ins = Qco2_rhs() # Make an instance of rhs
        Qh2o_rhs_ins = Qh2o_rhs() # Make an instance of rhs

        symb_time_units = C1_rhs_ins.CheckSymbTimeUnits(C1_rhs_ins)
        # Genetare the director
        RHS_list = [C1_rhs_ins, V1_rhs_ins, T1_rhs_ins, T2_rhs_ins, Qgas_rhs_ins, 
                        Qco2_rhs_ins, Qh2o_rhs_ins]
        self.MergeVarsFromRHSs(RHS_list, call=__name__)
        self.AddModule('Module1', Module1(C1=C1_rhs_ins, V1=V1_rhs_ins, T1=T1_rhs_ins, 
                        T2=T2_rhs_ins, Qgas=Qgas_rhs_ins, Qco2=Qco2_rhs_ins), Qh2o=Qh2o_rhs_ins)
        self.sch = ['Module1']

    def reset(self):
        self.Vars['T1'].val   = T1_in # np.random.RandomState().normal(21, 2)
        self.Vars['T2'].val   = T2_in # np.random.RandomState().normal(21, 2)
        self.Vars['V1'].val   = V1_in
        self.Vars['C1'].val   = C1_in # np.random.RandomState().normal(500, 1)
        self.Vars['Qgas'].val = 0
        self.Vars['Qco2'].val = 0
        self.Vars['Qh2o'].val = 0

    def update_controls(self, U=np.ones(11)):
        for i in range(len(U[0:10])):
            self.Vars['U'+str(i+1)].val = U[i]
        T1 = self.V('T1')
        self.Vars['I3'].val = T1 + U[10]*(T_cal-T1) 