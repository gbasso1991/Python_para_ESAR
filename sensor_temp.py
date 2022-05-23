#%%sensor_temp.py
# Giuliano Basso
import serial as ser
import serial.tools.list_ports
import numpy as np
import os 
import functools
import time
from datetime import datetime, timedelta
import serial
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
#%% Funciones de comunicacion con el sensor

def getHelp(serial_port):
    '''Printea el menu de ayuda'''
    serial_port.write(b'h\r')
    time.sleep(0.1)
    while serial_port.in_waiting>0:
        recentPacket = serial_port.readline()
        recentPacketString = recentPacket.decode('utf-8','ignore')#.rstrip('\n*')
        print(recentPacketString)
        time.sleep(0.1)

def getTemp(serial_port,channel=10):
    '''Printea la temperatura del canal especificado (default: 1)'''
    commandstr = 't'+str(channel)+'\r'
    serial_port.write(commandstr.encode('utf-8'))
    time.sleep(0.1)
    recentPacket = serial_port.readline()
    recentPacketString = recentPacket.decode('utf-8','ignore')
    temperature = float(recentPacketString.rstrip())
    #print(temperature)
    serial_port.reset_input_buffer()
    return temperature

def getTimeTemp(serialObj,t_0):
    '''Toma un objeto Serial y un tiempo inicial.
    Se comunica con el sensor y lee respuesta
    Devuelve 3 listas: Temperatura, Fecha, tiempo absoluto'''
    temp_array=[]
    date_array = []
    time_array =[]
    
    #Loop para obtener temperaturas cada 1 s
    while True:
        try:
            Temp = getTemp(serialObj)
            t= datetime.now()
            dt = datetime.now() - t_0 
            print(Temp,'-',t,'-',dt)
            temp_array.append(Temp)
            date_array.append(t.strftime('%H %M %S %f'))
            time_array.append(dt.total_seconds())
            time.sleep(0.2)
            
            fig = plt.figure(figsize=(8,7))
            fig.add_subplot(111)
            plt.plot(time_array,temp_array,'.-')
            plt.grid()
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Temperature vs t')
            plt.ylabel('Temperatura (ºC)')
            plt.xlabel('t (s)')
            plt.pause(0.2)
        
        except KeyboardInterrupt:
            fecha_salvado= datetime.today().strftime('%Y_%m_%d_%H%M%S')
            last_figure = plt.figure(figsize=(8,7))
            last_figure.add_subplot(111)
            plt.plot(time_array,temp_array,'.-')
            plt.grid()
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Temperatura vs. t')
            plt.ylabel('Temperatura (ºC)')
            plt.xlabel('t (s)')
            plt.savefig(f'registro_T_vs_t_'+ str(fecha_salvado)+'.png',dpi=300, facecolor='w')
            serialObj.close()
            print(f'Puerto serie {serialObj.name} cerrado')
            break
            
    return temp_array,date_array,time_array,fecha_salvado,last_figure       

#%% leo Puertos
ports_detected = serial.tools.list_ports.comports(include_links=False)
port_names=[]

for port in ports_detected:
    port_names.append(port.name)
    print('Puertos detectados:')
    print('-'*40) 
    print('Device: ',port.device)
    print('Name: ',port.name)
    print('Descritption: ',port.description)
    print('hwid: ',port.hwid)

for index,port in enumerate(ports_detected):
    '''Loop para eliminar puertos extra detectados con Linux'''
    if 'USB' not in port.device:
        #print('Puerto eliminado: ',index,port)
        ports_detected.remove(port)

# %% Elijo el 0 que en windows es el unico que detecta
pserie = ser.Serial(port=ports_detected[0].device,baudrate= 9600,stopbits=1,timeout=0)
if pserie.is_open:
    print(f'Puerto serie {pserie.name} abierto')
else:
    print('puerto cerrado')
# %%

#getHelp(pserie)

#%% Loop de adquisicion

t_0 = datetime.now()
temperatura,fecha,tiempo,fecha_salvado,figura = getTimeTemp(pserie,t_0=t_0)

#Encabezado del archivo de salida 
encabezado_salida = ['t (s)','Temperatura (°C)'] 
col_0 = tiempo
col_1 = temperatura

#Armo la tabla    
salida = Table([col_0, col_1]) 
formato_salida = {'t (s)':'%12.6f','Temperatura (°C)':'%.1f'} 

salida.meta['comments'] = [fecha_salvado]

ascii.write(salida,'registro_T_vs_t_'+ str(fecha_salvado) +'.txt',
            names=encabezado_salida,
            overwrite=True,
            delimiter='\t',
            formats=formato_salida)

#%%
if pserie.is_open:
   print(f'Puerto serie {pserie.device} abierto') 



#%% calibracion
'''Force temperature procedure:
        I. Apply a stable and known temperature to the sensor tip
        II. Check the display reading for abnormal deviation from the known temperature
        III. Send the “f” command followed by channel number, a blank character and the reference
        temperature value (example “f2 27.0 \r“). Temperatures must be entered in units as
        specified by the “u” command
        IV. Wait a few seconds
        V. Confirm that the readings correspond to the known temperature. '''
# =============================================================================
# 
# pserie.write(b'f1 18.4 \r')
# 
# 
# =============================================================================
# =============================================================================
# 
# data =  np.loadtxt(fname='registro_T_vs_t_2022_05_23_140256.txt',skiprows=2,delimiter='\t')
# t =data[:,0]
# temp =data[:,1]
# plt.plot(t,temp,'.-')
# 
# =============================================================================


