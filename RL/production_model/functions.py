from numpy import exp, floor, clip, arange, append

"""
*******  Note, the original TF is quite unsensitive to PA_mean ******
def TF( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    ### ORIGINAL Floration rate. DOES NOT DEPEND MUCH ON PA_MEAN!!
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-(1*k1_TF+PA_mean)/(2*k1_TF)))*k3_TF * Dt
"""

def TF_tmp( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    """Floration rate. TEMPORARY ... k1_TF = 150 the original below look wrong"""
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-PA_mean/k1_TF))*k3_TF * Dt

def TF( k1_TF, k2_TF, k3_TF, PA_mean, T_mean, Dt):
    ### ORIGINAL Floration rate. DOES NOT DEPEND MUCH ON PA_MEAN!!
    return (-0.75*k2_TF + 0.09*T_mean)*(1-exp(-(1*k1_TF+PA_mean)/(2*k1_TF)))*k3_TF * Dt


def f( k1_TF, k2_TF, PA_mean, T_mean, Dt):
    return -(1*k1_TF+PA_mean)/(2*k1_TF)

def Y_pot(k2_TF, C_t, B, D, M, X, T_mean):
    """Growth potential of each fruit."""
    return (T_mean - 10*k2_TF) *\
        B * M * exp(B * ( X - C_t))/(1 + D * exp(B * (X - C_t)))**( 1 + 1/D)

def Y_pot_veg(k2_TF, a, b, T_mean):
    """Growth potencial of the vegetative part"""
    return a*k2_TF + b*T_mean

def t_wg( dw_ef, A, f_wg):
    """Growth rate."""
    return f_wg * A / dw_ef

def f_wg( Y_pot, Y_sum):
    """Sink stregth, without kmj."""
    return Y_pot / Y_sum  ### No units
