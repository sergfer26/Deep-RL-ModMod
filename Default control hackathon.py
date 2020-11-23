# -*- coding: utf-8 -*-
"""
Adjust the following code to set the control variables dyamically

Use Python Programing Lenguage syntax
"""

# Do not modify following function (Adjust_Setpoints)
def Adjust_Setpoints(CO2Air,CO2Top,RHAir,RHTop,TAir,TCan,TCovE,TCovI,TFlr,Tpipe,TSoil1,TSoil2,TSoil3,TSoil4,TSoil5,TThrScr,TTop,VPAir,VPCan,VPTop,IGlob,TOut,TSky,TSoOut,RHOut,VPOut,CCov,WSpd,Yr,Mth,MthD,Hr,min,WkD,YrD,CtrlStep):


# Code that determines Day/Night Setpoints
    if IGlob < 5:
        HeatSet = 25
        VentSet = 28
        VPDSet = 1.8
    else:
        HeatSet = 25
        VentSet = 28
        VPDSet = 1.8

# Code that turns ON/OFF the Lamps
    if IGlob < 700 and Hr > 7 and Hr < 19: 
        Lamp = 1
    else:
        Lamp = 0


# Code that sets the CO2 enrichment setpoint
    if IGlob > 5 or Lamp > 0:
        CO2Set = 400
    else:
        CO2Set = 400


# Code that prevents a VPD too low by setting a minimum window opening
    if VPCan-VPAir < 0.5: 
        MinLee = 0.02
    else:
        MinLee = 0


# Code that controls the thermal screen
    if IGlob > 5 and IGlob < 200 and TOut < 12:
        ThScrSet = 0.5
    elif IGlob < 5:
        ThScrSet = 1
    else:
        ThScrSet = 0


# Code that determines season setpoints
    if YrD > 80 and YrD < 172:       #"Spring"
        Flowers = 0.5
    elif YrD > 172 and YrD < 264:    #"Summer"
        Flowers = 0.5
    elif YrD > 264 and YrD < 355:    #"Autumn"
        Flowers = 0.5
    else:                           #"Winter"
        Flowers = 0.5

    MinRail = 20
    MinCrop = 20
    MinWind = 0
    InBlkScrSet = 0
    ExBlkScrSet = 0
    WtW = 0
    WtrInt = 0
    WtrDur = 0
    Leaves = 25
    Stems = 1.6
    Dcap = 0

    SetPoints = [HeatSet,VentSet,MinRail,MinCrop,MinLee,MinWind,VPDSet,ThScrSet,InBlkScrSet,ExBlkScrSet,WtW,Lamp,CO2Set,WtrInt,WtrDur,Flowers,Leaves,Stems,Dcap]
    return(SetPoints)