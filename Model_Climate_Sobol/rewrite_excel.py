import pandas as pd
import numpy as np 
import xlwt
from preparacion_datos import filter_by_5_min, repeat_data

index = 1

xls = pd.read_excel('Read_Data.xls', sheet_name=None)
samples = xls['Climate1'].shape[0] # size 33133 - > approx 57.5 days of data

data = pd.read_csv('index_'+ str(index) +'.csv')
data = filter_by_5_min(data)
reps = 58 
data = repeat_data(data, reps)

xls['Climate1']['Tair'] = np.array(data['T2M'][0 :samples])# replace with data simulated
xls['Climate2']['CO2air'] = np.array(data['C1M'][0 : samples])

with pd.ExcelWriter('Read_Data_index_'+ str(index) +'.xls', engine='xlwt') as writer:
    for sheet in xls.keys(): # sheet names
        xls[sheet].to_excel(writer, sheet_name=sheet)

writer.save()
writer.close()
