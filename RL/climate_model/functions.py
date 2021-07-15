from numpy import sqrt, exp 

def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*(U1*f5 + (1-U1)*f5) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*(U1*f7 + (1-U1)*f7) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt(max((omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0))


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)

### functions of this RHS ###


def kappa4(phi2):
    return phi2


def o1(eta13, h6):
    return eta13*h6


def o2(U10, psi2, alpha6):
    return U10 * psi2 / alpha6


def o3(C1, I10, f1):
    return f1 * (C1 - I10)


def o4(I11, I12, I13, psi3):
    return psi3*I11*(I12 - I13)


def o5(C1, I10, f2, f3, f4):
    return (C1 - I10) * (f2 + f3 + f4)

def o6 (C1, omega3):
    return omega3*C1

######### V1 ##############

### previous functions ###


def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*( U1*f5 + (1-U1)*f5 ) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*( U1*f7 + (1-U1)*f7 ) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt( max( (omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0 ) )


def h3(T2, V1, U3, I6, lamb1, lamb2, alpha6, gamma2, q6):
    num = (U3*lamb1*lamb2/alpha6)*(I6-T2)
    den = T2 - I6 + ( 6.4e-9 * gamma2*(V1 - q6 ) )
    return num/den


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)


def q1(I1, rho3, alpha5, gamma, gamma2, gamma3, q3):
    return (2*rho3*alpha5*I1) / ( gamma*gamma2*(gamma3 + q3 ) )


def q2(T1):
    try:
        if (T1 > 0):
            return 0.61121*exp((18.678 - (T1/234.5)) * (T1/(257.14+T1)))
        else:
            return 0.61115*exp((23.036 - (T1/333.7)) * (T1/(279.82+T1)))
    except:
        return 14   # Esta opción es por si ocurre un error numérico. 14 es el valor inicial de V1


def q3(I14, gamma4, q4, q5, q10):
    return gamma4*q10*q4*q5


def q4(C1, eta4, q8):
    return 1 + q8*((eta4*C1 - 200)**2)


def q5(V1, q2, q9):
    return 1 + q9*((q2 - V1)**2)


def q6(I6):
    if (I6 > 0):
        return 0.61121*exp((18.678 - (I6/234.5)) * (I6/(257.14+I6)))
    else:
        return 0.61115*exp((23.036 - (I6/333.7)) * (I6/(279.82+I6)))


def q7(I14, delta1, gamma5):
    return (1 + exp(gamma5*(I14 - delta1)))**-1


def q8(delta4, delta5, q7):
    return delta4*(1 - q7) + delta5*q7


def q9(delta6, delta7, q7):
    return delta6*(1 - q7) + delta7*q7


def q10(I14, delta2, delta3):
    return (I14 + delta2)/(I14 + delta3)

### functions of this RHS ###


def kappa3(T2, psi1, phi2, omega2):
    return (psi1*phi2) / (omega2*(T2+273.15))


def p1(V1, q1, q2):
    return q1*(q2 - V1)


def p2(rho3, eta5, phi5, phi6, f1):
    return rho3*f1*(eta5*(phi5 - phi6) + phi6)


def p3(U9, phi9, alpha6):
    return U9*phi9/alpha6


def p4(eta12, h6):
    return eta12*h6


def p5(T2, V1, I5, psi1, omega2, f2, f3, f4):
    return (psi1/omega2)*((V1/(T2+273.15)) - (I5/(I5+273.15)))*(f2 + f3 + f4)


def p6(T2, V1, psi1, omega2, f1):
    return f1*(psi1/omega2)*(V1/(T2+273.15))


def p7(V1, h3, q6):
    if (V1 < q6):
        return 0
    else:
        return (6.4e-9)*h3*(V1 - q6)


######### T1 ##############
#### Symbols
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm')  # Symbolic use of base phisical units

### previous functions ###


def a1(I1, beta3):
    return 1 - exp(-beta3*I1)


def b1(U1, tau3):
    return 1 - U1*(1 - tau3)


def r2(I1, beta1, rho1, r4):
    return r4*(1 - rho1)*( 1 - exp(-beta1*I1) )


def r3(I1, beta1, beta2, rho1, rho2, r4):
    return r4*exp(-beta1*I1)*rho2*(1 - rho1)*(1 - exp(-beta2*I1))


def r4(I2, eta1, eta2, tau1):
    return (1 - eta1)*tau1*eta2*I2


def g1(a1):
    return 0.49*a1


def g2(tau2, b1):
    return tau2*b1


def p1(V1, q1, q2):
    return q1*( q2 - V1 )


def q1(I1, rho3, alpha5, gamma, gamma2, gamma3, q3):
    return (2*rho3*alpha5*I1) / ( gamma*gamma2*( gamma3 + q3 ) )


def q2(T1):
    try:
        if (T1 > 0):
            return 0.61121*exp((18.678 - (T1/234.5)) * (T1/(257.14+T1)))
        else:
            return 0.61115*exp((23.036 - (T1/333.7)) * (T1/(279.82+T1)))
    except:
        return 14


def q3(I14, gamma4, q4, q5, q10):
    return gamma4*q10*q4*q5


def q4(C1, eta4, q8):
    return 1 + q8*((eta4*C1 - 200)**2)


def q5(V1, q2, q9):
    return 1 + q9*((q2 - V1)**2)


def q7(I14, delta1, gamma5):
    return (1 + exp(gamma5*(I14 - delta1)))**-1


def q8(delta4, delta5, q7):
    return delta4*(1 - q7) + delta5*q7


def q9(delta6, delta7, q7):
    return delta6*(1 - q7) + delta7*q7


def q10(I14, delta2, delta3):
    return (I14 + delta2)/(I14 + delta3)

### functions of this RHS ###


def kappa1(I1, alpha1):
    return alpha1*I1


def r1(r2, r3):
    return r2 + r3


def r5(I2, alpha2, eta1, eta3):
    return (1 - eta1)*alpha2*eta3*I2


def r6(T1, I3, alpha3, epsil1, epsil2, lamb, g1):
    return alpha3*epsil1*epsil2*g1*lamb*((I3 + 273.15)**4 - (T1 + 273.15)**4)


def h1(T1, T2, I1, alpha4):
    return 2*alpha4*I1*(T1-T2)


def l1(gamma2, p1):
    return gamma2*p1


def r7(T1, I4, epsil2, epsil3, lamb, a1, g2):
    return a1*epsil2*epsil3*g2*lamb*((T1 + 273.15)**4 - (I4 + 273.15)**4)


######### T2 ##############
#### Symbols ########
mt, mg, m, C, s, W, mg_CO2, J, Pa, kg_water, kg, K, ppm, m_cover, kg_air = symbols('mt mg m C s W mg_CO2 J Pa kg_water kg K ppm m_cover kg_air')  # Symbolic use of base phisical units

### previous functions ###


def b1(U1, tau3):
    return 1 - U1*(1 - tau3)


def f1(U2, phi7, alpha6):
    return U2*phi7/alpha6


def f2(U1, eta6, eta7, eta8, f5, f6):
    if (eta7 >= eta8):
        return eta6*f5 + 0.5*f6
    else:
        return eta6*(U1*f5 + (1-U1)*f5) + 0.5*f6


def f3(U7, phi8, alpha6):
    return U7*phi8/alpha6


def f4(U1, eta6, eta7, eta8, f6, f7):
    if (eta7 >= eta8):
        return eta6*f7 + 0.5*f6
    else:
        return eta6*(U1*f7 + (1-U1)*f7) + 0.5*f6


def f5(I8, alpha6, n1, n2, n3):
    return n1*n2*I8*sqrt(n3) / (2*alpha6)


def f6(I8, nu4):
    if (I8 < 0.25):
        return 0.25*nu4
    else:
        return nu4*I8


def f7(T2, U8, I5, I8, nu5, alpha6, omega1, nu6, n1, n3):
    T_bar = (T2 + I5)/2
    return (U8*nu5*n1)/(2*alpha6) * sqrt(max((omega1*nu6*(T2-I5))/(2*(T_bar+273.15)) + n3*I8, 0))


def g3(I1, beta3, gamma1, phi1, tau1, b1):
    return tau1*b1*(1 - 0.49*pi*gamma1*phi1)*exp(-beta3*I1)


def g4(U1, tau2):
    return U1*tau2


def h8(T2, I5, I8, alpha6, lamb5, lamb6, lamb7, lamb8):
    return lamb5*(lamb6 + lamb7*(I8**lamb8))*(T2 - I5)/alpha6


def h9(T2, I5, alpha5, rho3, f4):
    return rho3*alpha5*f4*(T2 - I5)


def n1(U5, nu1, eta10):
    return nu1*(1 - eta10*U5)


def n2(U6, nu3):
    return U6*nu3


def n3(U5, nu2, eta11):
    return nu2*(1 - eta11*U5)


def q6(I6):
    if (I6 > 0):
        return 0.61121*exp((18.678 - (I6/234.5)) * (I6/(257.14+I6)))
    else:
        return 0.61115*exp((23.036 - (I6/333.7)) * (I6/(279.82+I6)))


def r9(I2, alpha8, alpha9, eta2, eta3):
    return (alpha8*eta2 + alpha9*eta3)*I2


def r11(T2, I4, lamb, epsil3, epsil4, g3):
    return epsil4*epsil3*g3*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)


def r12(T2, I4, lamb, epsil3, epsil5, g4):
    return epsil5*epsil3*g4*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)


def r13(T2, I4, lamb, epsil3, epsil6):
    return epsil6*epsil3*lamb*((T2 + 273.15)**4 - (I4 + 273.15)**4)

### functions of this RHS ###


def kappa2(alpha5, rho3, phi2):
    return phi2*rho3*alpha5


def h1(T1, T2, I1, alpha4):
    return 2*alpha4*I1*(T1-T2)


def h2(I5, alpha5, gamma2, eta5, rho3, phi5, phi6, f1):
    return f1*(rho3*alpha5*I5 - gamma2*rho3*(eta5*(phi5 - phi6)))


def h3(T2, V1, U3, I6, lamb1, lamb2, alpha6, gamma2, q6):
    num = (U3*lamb1*lamb2/alpha6)*(I6-T2)
    den = T2 - I6 + (6.4e-9 * gamma2*(V1 - q6))
    return num/den


def h4(T2, I3, gamma1, phi1):
    return (1.99*pi*phi1*gamma1*(abs(I3 - T2)**0.32))*(I3 - T2)


def h5(T2, I7, lamb3):
    return lamb3*(I7 - T2)


def h6(U4, lamb4, alpha6):
    return U4*lamb4/alpha6


def r8(I2, alpha2, alpha7, eta1, eta2, eta3, tau1, r9):
    return eta1*I2*(tau1*eta2 + (alpha2 + alpha7)*eta3) + r9


def h7(T2, I5, alpha5, rho3, f2, f3, h8, h9):
    return rho3*alpha5*(f2 + f3)*(T2 - I5) + h8 + h9


def h10(T2, alpha5, rho3, f1):
    return f1*rho3*alpha5*T2


def l2(U9, alpha6, gamma2, phi9):
    return gamma2*U9*phi9/alpha6


def r10(r11, r12, r13):
    return r11 + r12 + r13


def h11(T2, I7, nu7, nu8, phi2):
    return 2*nu7*(T2 - I7)/(phi2 + nu8)

def H_Boil_Pipe(r6,h4):
    return max(r6 + h4,0)