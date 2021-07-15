import pandas as pd
import numpy as np


def Mes(date):
    '''Regresa el mes'''
    return date[0:2] 

def Dia(date):
    '''Regresa el dia'''
    return date[3:5] 

def Indexes(data, season):
    '''
    Season 1: Jan - Apr
    Season 2: Apr - Agu
    '''
    if season == 1: #Primera semana de enero a primera de abril
        season = ['0'+str(i) for i in range(1,5)]
        week = ['0'+str(i) for i in range(1,8)]
    if season == 2: #Primera semana de abril a mediados de agosto
        season = ['0'+str(i) for i in range(1,5)]
        week = ['0'+str(i) if i <10 else str(i) for i in range(1,15)]
    indexes = list()
    for i,fecha in enumerate(data['Date']):
        mes = Mes(fecha)
        if mes in season:
            if mes == season[-1]:
                if Dia(fecha) in week:
                    indexes.append(i)
            else:
                indexes.append(i)
    return indexes

