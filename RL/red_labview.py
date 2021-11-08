import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable


def NNcontrol(CO2Air,CO2Top,RHAir,RHTop,TAir,TCan,TCovE,
    TCovI,TFlr,Tpipe,TSoil1,TSoil2,TSoil3,TSoil4,TSoil5,
    TThrScr,TTop,VPAir,VPCan,VPTop,IGlob,TOut,TSky,TSoOut,
    RHOut,VPOut,CCov,WSpd,Yr,Mth,MthD,Hr,min,WkD,YrD,CtrlStep,h,n):
    '''
    Entradas que usamos:
    CO2Air, RHAir, TCan, IGlob, 
    Faltaria:
    - Incremento del peso de lo cosechado en las ultimas 24h (h)
    - Número de frutos cosechados en las ultimas 24h  (n)
    Salidas
    U1  = control de la pantalla térmica. Valores entre [0,1]  
    U4  = calentador de aire del invernadero. Valores entre [0,1] (?)
    U8  = respiraderos del techo Valores entre [0,1]  
    U9  = sistema de niebla. Valores entre [0,1]  
    U10 = fuente externa del CO2. Valores entre [0,1]  
    '''
    NN = torch.jit.load("model_64_64_64")
    x = [CO2Air, RHAir, TCan, IGlob, h, n]
    x = torch.from_numpy(np.array(x)).float().unsqueeze(0)
    controles = NN.forward(x) #[U1,U4,U8,U9,U10]
    controles = controles.detach().numpy()[0,:]
    return controles
