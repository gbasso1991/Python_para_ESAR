#%%sensor_temp.py
# Giuliano Basso
import serial as sr
import serial.tools.list_ports
import numpy as np
import os 
import functools
import time
from datetime import datetime, timedelta
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#%% Comunicacion 

def getHelp(serial_port):
    '''Le pide al sensor que printee el menu de ayuda'''
    serial_port.write(b'h\r')
    time.sleep(0.1)
    while serial_port.in_waiting>0:
        recentPacket = serial_port.readline()
        recentPacketString = recentPacket.decode('utf-8','ignore')#.rstrip('\n*')
        print(recentPacketString)
        time.sleep(0.1)

def getTemp(serial_port,channel=1):
    '''Comunica al sensor el comando para obtener la temperatura del canal especificado'''
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
            time.sleep(0.89)
            
            fig = plt.figure()
            fig.add_subplot(111)
            plt.plot(time_array,temp_array,'o-')
            plt.grid()
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Temperature vs t')
            plt.ylabel('Temperatura (ºC)')
            plt.xlabel('t (s)')
            plt.pause(0.2)
        
        except KeyboardInterrupt:
            last_figure = plt.figure()
            last_figure.add_subplot(111)
            plt.plot(time_array,temp_array,'o-')
            plt.grid()
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Temperature vs t')
            plt.ylabel('Temperatura (ºC)')
            plt.xlabel('t (s)')
            plt.show()
            break
            
    return temp_array,date_array,time_array,last_figure       

#%% leo Puertos
ports_detected = serial.tools.list_ports.grep(regexp='USB')
ports_detected_2 = serial.tools.list_ports.comports()

# %
for port in ports_detected_2:
    print(port.device)
    print(port.name)
    print(port.description)
    print(port.hwid)
    print('-'*20) 

# %% Elijo el 0 que en windows es el unico que detecta
pserie = sr.Serial(port='COM4',baudrate= 9600,stopbits=1,timeout=0)
if pserie.is_open:
    print(f'Puerto serie {pserie.name} abierto')
else:
    print('puerto cerrado')
# %%

getHelp(pserie)
#%%
getTemp(pserie)
#%%
t_0 = datetime.now()
temperatura,fecha,tiempo,figura = getTimeTemp(pserie,t_0=t_0)
 


#%%
figura
#%%
pserie.close()
# %%

