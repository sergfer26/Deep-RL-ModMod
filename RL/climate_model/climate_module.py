from ModMod import Module
from .constants import OTHER_CONSTANTS
from scipy.stats import norm
RANDOM = OTHER_CONSTANTS['model_noise'].val

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
        #breakpoint()
        if RANDOM:
            T1r = self.V('T1') + norm.rvs(scale=s1)
            T2r = self.V('T2') + norm.rvs(scale=s1)
            V1r = self.V('V1') + norm.rvs(scale=s2)
            C1r = self.V('C1') + norm.rvs(scale=s3)
            
        else:
            T1r = self.V('T1') 
            T2r = self.V('T2') 
            V1r = self.V('V1') 
            C1r = self.V('C1') 
        # Actualización de las variables
        self.V_Set('T1', T1r)
        self.V_Set('T2', T2r)
        self.V_Set('V1', V1r)
        self.V_Set('C1', C1r)
        # Avance del RHS
        self.AdvanceRungeKutta(t1)
        return 1
