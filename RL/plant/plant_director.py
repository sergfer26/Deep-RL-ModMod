from ModMod import Director

#RHS del CO2 intracelular   
from .photosynthesis_model.ci_rhs import Ci_rhs

#Módulo de crecimiento para una planta
from production_model.q_rhs import Q_rhs

from .plant_module import Plant

#Director esclavo de una planta 
def PlantDirector( beta, return_Q_rhs_ins=False):
    """Build a Director to hold a Plant, with beta PAR parameter."""

    ### Start model with empty variables
    Dir = Director( t0=0.0, time_unit="", Vars={}, Modules={} )
    
    ### Add the photosynthesis module:
    Ci_rhs_ins = Ci_rhs()
    
    Dir.AddTimeUnit( Ci_rhs_ins.GetTimeUnits())

    Dir.MergeVarsFromRHSs( Ci_rhs_ins, call=__name__)
    
    ### Start new plant  rhs
    Q_rhs_ins = Q_rhs()

    Dir.AddTimeUnit( Q_rhs_ins.GetTimeUnits())

    Dir.MergeVarsFromRHSs( Q_rhs_ins, call=__name__)
    
    ### Add an instance of Module 1:
    Dir.AddModule( "Plant", Plant( beta, Q_rhs_ins, Ci_rhs_ins) )
    Dir.sch = [ "Plant" ]

    if return_Q_rhs_ins:
        return Dir, Q_rhs_ins
    else:
        return Dir
