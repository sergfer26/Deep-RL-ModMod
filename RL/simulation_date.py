import pandas as pd
from dateutil.parser import parse
from datetime import datetime
from trainRL import sim 


def get_index(data, date):
    '''
    data = dataframe Inputs_Bleisweik.csv
    date = datetime.datatime(y, m, d, h)
    '''
    data['Date'] = data['Date'].apply(lambda x: parse(x))
    idx = data[data['Date'] == date].index.values
    return int(idx)
    
