import pandas as pd
def main():
    data_inputs = pd.read_csv('Inputs_Bleiswijk.csv')
    wind_speed  = data_inputs.I8[0]
    if wind_speed == 6.77:
        print('La velocidad del viento esta en Km x h**-1, se va a convertir')
        #[Km/h] = [1000m/1Km]*[1h/3600s] 
        wind_speed     = data_inputs.I8/3.6
        data_inputs.I8 = wind_speed
        data_inputs.to_csv('Inputs_Bleiswijk.csv',index=0)
    else:
        print('Ya esta en las unidades correctas, no voy a hacer nada')


if __name__ == '__main__':
    main()