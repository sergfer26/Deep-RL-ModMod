from solver_climate import Sol_Climate
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform

theta = np.array([2000, 15, 1.9e5]) # Init0    
Init1 = np.array([4000, 25, 2.7e5])
K = 1440 * 7

C1_in = 100
V1_in = 30
T2_in = 50
T1_in = 40

S = [C1_in, V1_in, T1_in, T2_in]

def ClimateU(A, k=K):
    """
    Esta es la función energía:
    U = - log [ f (theta | Datos) ].
    theta = [alpha1, phi2, psi2].    
    """
    ## Se soluciona el sistema para nmrec minutos
    # k = 2*1440
    k = K
    Res = Sol_Climate(A, theta=theta, S=S, nmrec=k)
    C1M = Res[0]
    V1M = Res[1]
    T1M = Res[2]
    T2M = Res[3]
    ## Cálculo de la energía
    return C1M, V1M, T1M, T2M

def grafica(X1, X2):
    x = np.linspace(0,1, K)
    C1M_x1,V1M_x1,T1M_x1,T2M_x1 = X1
    C1M_x2,V1M_x2,T1M_x2,T2M_x2 = X2
    fig, ((c1, v1), (t1, t2)) = plt.subplots(2, 2)

    c1.plot(x, C1M_x1, c='r') 
    c1.plot(x, C1M_x2, c='b')
    c1.plot(x,C1_in * np.ones(K))
    
    v1.plot(x, V1M_x1, c='r') 
    v1.plot(x, V1M_x2, c='b')
    v1.plot(x,V1_in * np.ones(K))

    t1.plot(x, T1M_x1, c='r') 
    t1.plot(x, T1M_x2, c='b')
    t1.plot(x,T1_in * np.ones(K))

    t2.plot(x, T2M_x1, c='r') 
    t2.plot(x, T2M_x2, c='b')
    t2.plot(x,T2_in * np.ones(K))
    plt.show()    


# le quite normales en Solver/Advance 1350
A1 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
A2 = np.array(A1)
A2[3] = 0


X1 = ClimateU(A1)
#X2 = ClimateU(A2)
grafica(X1, X1)