from numpy import exp, floor, clip, arange, append
from ModMod import Module
import copy
from .photosynthesis_model.functions import *
from production_model.functions import * 

#Módulo de crecimiento para una planta
class Plant(Module):
    def __init__( self, beta, Q_rhs_ins, Ci_rhs_ins, Dt_f=0.1, Dt_g=1):  # Dt_f=0.1, Dt_g=0.25
        """Models one plant growth, with a variable number of fruits."""
        ## Dt_f is the photosynthesis Dt, this is use for RK advance
        ## Dt_g is the growth Dt, this is use for the update and anvance of the fruits
        super().__init__(Dt_f) #Time steping of module, days
        ### Always, use the super class __init__, there are several other initializations
        
        self.Dt_g = Dt_g
        self.beta = beta ## must be in (0,1]
    
        ### Time units= hours
        
        ### Vegetative part
        self.veget = [0.0 , 0.0] ## characteristics for vegetative part: Weight and growth potential 

        self.fruits = [] # No fruits
        self.n_fruits = 0 ## Current number of fruits
        self.n_fruits_h = 0 ## total number of fruits harvested
        self.new_fruit = 0  ## Cummulative number of fruits
        self.m = 4 ## Number of characteristics for each fruit: thermic age, weight, growth potential and Michaelis-Menten constant
        ### Module specific constructors, add RHS's

        self.AddStateRHS( 'Q', Q_rhs_ins)
        self.AddStateRHS( 'Ci', Ci_rhs_ins)

    def Advance( self, t1):
        """Update the plant/fruit growth. Update global variables, to time t1."""
        
        ### This creates a set of times knots, that terminates in t1 with
        ### a Deltat <= self.Dt
        tt = append(arange( self.t(), t1, step=self.Dt_g), [t1])
        #print(tt)
        steps = len(tt)
        for i in range( 1, steps):
            
            # Reset the assimilation rate
            rs = 0.0
            self.V_Set('A', rs)
            
            ## Assimilation rate update
            V_cmax1 = V_cmax( T_f=self.V('T'), V_cmax25=self.V('V_cmax25'), Q10_Vcmax=self.V('Q10_Vcmax'), k_T=self.V('k_T'), k_d=self.V('k_d') )
            R_d1 = R_d( V_cmax=V_cmax1 )
            tau1 = tau( T_f=self.V('T'), tau_25=self.V('tau_25'), Q10_tau=self.V('Q10_tau'), k_T=self.V('k_T') )
            K_C1 = K_C( T_f=self.V('T'), K_C25=self.V('K_C25'), Q10_KC=self.V('Q10_KC'), k_T=self.V('k_T') )
            K_O1 = K_O( T_f=self.V('T'), K_O25=self.V('K_O25'), Q10_KO=self.V('Q10_KO'), k_T=self.V('k_T') )
            Gamma_st1 = Gamma_st( T_f=self.V('T') )
            I_21 = I_2( I =self.V('PAR'), f=self.V('f'), ab=self.V('ab') )
            J1 = J( I_2=I_21, J_max=self.V('J_max'), theta=self.V('theta'), k_d=self.V('k_d') )
            A_R1 = A_R( O_a=self.V('O_a'), tau=tau1, C_i=self.V('Ci'), V_cmax=V_cmax1, Gamma_st=Gamma_st1, K_C=K_C1, K_O=K_O1, phi=self.V('phi') )
            A_f1 = A_f( C_i=self.V('Ci'), Gamma_st=Gamma_st1, J=J1, k_JV=self.V('k_JV') )
            A_acum1 = A_acum( V_cmax=V_cmax1 )
            A1 = abs( A( A_R=A_R1, A_f=A_f1, A_acum=A_acum1, R_d=R_d1, fc=self.V('fc') ) )
            self.V_Set('A', A1)
            
            ### Check if a fruit has reched is maximum thermic age, then harvest it
            fruits = copy.deepcopy(self.fruits) # copy of all fruits
            wk = 0.0
            nfk = 0
            for fruit in self.fruits: 
                if (fruit[0] > 275 or fruit[1]>360): # It is harvested when a fruit reaches a thermic age of 275 °C d or if the fruit's weigth is greater than 360 g
                    self.n_fruits -= 1 # number fruits in crop
                    self.n_fruits_h += 1 # accumulated number fruits harvested
                    w = self.V( 'Q_h') + fruit[1] # accumulated weight fruits harvested
                    self.V_Set( 'Q_h', w)
                    wk += fruit[1] # weigth fruits harvested in this moment
                    nfk += 1 # number fruits harvested in this moment
                    fruits.remove(fruit) # Harvested fruits are removed from the list
            self.fruits = fruits
            self.V_Set( 'n_k', nfk)
            self.V_Set( 'h_k', wk)
            ### With the Floration Rate, create new fruits
            PA_mean_i = self.beta * self.V('PAR')
            self.new_fruit += TF( k1_TF=self.V('k1_TF'), k2_TF=self.V('k2_TF'), k3_TF=self.V('k3_TF'),\
                    PA_mean=PA_mean_i, T_mean=self.V('T'), Dt=tt[i]-tt[i-1])
            new_fruit_n = self.new_fruit 
            if new_fruit_n >= 1:
                #nw = new_fruit_n
                nw = int(floor(self.new_fruit))
                for nf in range(nw):
                    ### Add new fruit
                    self.fruits += [[ 0.0, 0.0, 0.0, 0.0]] 
                    ### also the growth potential, as an auxiliar for calculations
                    self.n_fruits += 1 
                ### Leave the rational part of new_fruit
                self.new_fruit = max( 0, self.new_fruit - nw)
            
            ### Update thermic age of all fruits
            for fruit in self.fruits:
                fruit[0] += ( max( 0 , self.V('T') - 10 ) )* (tt[i]-tt[i-1]) ## Thermic age never decreases
            
            ### Update growth potencial for vegetative part
            self.veget[1] = self.V('a') + self.V('b')*self.V('T') 
            ### Update Growth potential and Michaelis-Menten constants of all fruits
            tmp = 0.0
            tmp1 = self.veget[1] / self.V('A') # start with the growth potencial of vegetative part
            for fruit in self.fruits:
                x = fruit[0] ## Thermic age
                ### Michaelis-Menten constants 
                if x <= self.V('C_t') :
                    fruit[3] = 0.05*tmp*(self.V('C_t') - x) / self.V('C_t')
                ### Growth potential
                fruit[2] = clip( Y_pot( k2_TF=self.V('k2_TF'), C_t=self.V('C_t'),\
                     B=self.V('B'), D=self.V('D'), M=self.V('M'), X=x, T_mean=self.V('T')),\
                     a_min=0, a_max=exp(300))
                tmp += fruit[2]
                tmp1 += fruit[2] / ( fruit[3] + self.V('A') )
            #self.V_Set( 'Y_sum', tmp)
            
            ### Update weight of vegetative part
            f_wg_veg =  self.veget[1] / ( self.V('A') * tmp1 ) # The sink strentgh of vegetative part
            self.veget[0] += t_wg( dw_ef=self.V('dw_ef_veg'), A=self.V('A'), f_wg=f_wg_veg) * (tt[i]-tt[i-1])
            #### Update weight of all fruits
            tmp2 = 0.0
            Dt = (tt[i]-tt[i-1])
            for fruit in self.fruits:
                f_wg =  fruit[2] / ( ( fruit[3] + self.V('A') ) * tmp1 ) # The sink strentgh of the fruit
                dwh = t_wg( dw_ef=self.V('dw_ef'), A=self.V('A'), f_wg=f_wg) * Dt # dry weight
                pdw = 0.023 # percentage of dry weight
                fruit[1] += dwh / pdw # Fresh weight  
                tmp2 += fruit[1] #Total weight
            
            #### Update assimilation rate after distribution
            m = ( f_wg_veg / self.V('dw_ef_veg') ) + ( (1 - f_wg_veg ) / self.V('dw_ef') )
            As = self.V('A')*( 1 - self.V('a_ef')*Dt*m ) # A = A - ( Total weigth of fruits and vegetative part )
            self.V_Set('A', As )  
            
            #### Total weight of the fruits
            self.V_Set( 'Q', tmp2)
            
            #### Advance of the RHS
            self.AdvanceAssigment(t1) # Set Q
            self.AdvanceRungeKutta(t1) # Set A
            
        return 1
