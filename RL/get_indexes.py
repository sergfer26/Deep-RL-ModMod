import pandas as pd
import datetime

def Mes(date,month):
    list_ = list(date)
    return list_[0] + list_[1] == month

def Indexes(data,month):
    Filter = [Mes(x,month) for x in data['Date']]
    return [i for i, j in enumerate(Filter) if j == True]


