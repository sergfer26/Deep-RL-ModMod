import pandas as pd
import numpy as np 
import xlwt
import pathlib
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt


class Struct_var:
    def __init__(self, sheet, variable, index):
        self.sheet = sheet
        self.variable = variable
        self.index = index

def xls_to_csv():
    file_name = 'Read_Inputs_Inf.xls'
    Vars = [Struct_var('Meteo','Iglob','I2'), Struct_var('Control','HeatTemp_Vip','I3'), \
    Struct_var('Meteo','Tout','I4'), Struct_var('Meteo','Tout','I5'), Struct_var('Control','VentLeew_Vip','I6'), \
    Struct_var('Meteo','Windsp','I8'),Struct_var('Meteo','Iglob','I14'), Struct_var('Climate1','RHair','RH'), \
    Struct_var('Meteo','PARout','PAR')]

    xls = pd.read_excel(file_name, sheet_name=None)
    dic = {var.index: xls[var.sheet][var.variable] for var in Vars}
    data = pd.DataFrame(dic)
    data = data.dropna(how='any')
    data.to_csv('Inputs.csv',index = False)


def csv_to_csv():
    file_name = 'Historic_Weather_Data_Bleiswijk_TS.csv'
    data = pd.read_csv(file_name)
    data.columns = ['Time_Stamp','Year','Month','Day','Hour','Minute' \
                    ,'I5','RH','Total_Cloud_Cover','High_Cloud_Cover' \
                    ,'Medium_Cloud_Cover','Low_Cloud_Cover','I2','I8'\
                    ,'Wind_Direction','Wind_Speed80','Wind_Direction80'\
                    ,'Wind_Speed900','Wind_Direction900']
    dates = []
    for column in ['Year', 'Month', 'Day', 'Hour']: 
        data[column] = pd.to_numeric(data[column], downcast='integer')
        
    for row in range(len(data)):
        dates.append(datetime(data['Year'][row],data['Month'][row],data['Day'][row], data['Hour'][row]).strftime(r"%x %H"))
 
    data['Date'] = dates
    data['I14'] = data['I2'] # Por ahora son el mismo (PAR)

    data[['I2','I5', 'I8', 'I14','RH', 'Date' ]].to_csv('Inputs_Bleiswijk.csv',index = False)

if __name__ == '__main__':
    csv_to_csv()
