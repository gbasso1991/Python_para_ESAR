#%%sensor_temperatura.py
#https://realpython.com/python-encodings-guide/
from tkinter import *
import serial.tools.list_ports
import functools
import time

ports = serial.tools.list_ports.comports()#chekeo que puertos detecta la pc
serialObj = serial.Serial() #creo instancia

root = Tk() 
root.config(bg='grey')

def initComPort(index):
    currentPort=str(ports[index])
    print(currentPort)
    comPortVar= str(currentPort.split(' ')[0])
    serialObj.port= comPortVar
    serialObj.baudrate= 9600
    serialObj.open()

for onePort in ports:
    comBotton= Button(root,text=onePort,font=('Calibri','13'),height=1,width=40,command= functools.partial(initComPort,index=ports.index(onePort)) )
    comBotton.grid(row=ports.index(onePort),column=0)

dataCanvas = Canvas(root,width=600,height=400,bg='white')
dataCanvas.grid(row=0,column=1,rowspan=100)

vsb = Scrollbar(root,orient='vertical',command= dataCanvas.yview)#barra de desplazamiento vertical
vsb.grid(row=0,column=2,rowspan=100,sticky='ns')

dataCanvas.config(yscrollcommand=vsb.set)
dataFrame= Frame(dataCanvas,bg='white')
dataCanvas.create_window((10,0),window=dataFrame,anchor='nw')


def checkSerialPort():
    if serialObj.isOpen():
        serialObj.write(b't1\r')
        
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf-8','strict').rstrip('\n*')
        
        Label(dataFrame,text=recentPacketString,font=('Calibri','13'),bg='white').pack() 
        #time.sleep(0.1)
    else:
        pass

while True:

    checkSerialPort()
    root.update()
    dataCanvas.config(scrollregion=dataCanvas.bbox('all'))
#%% Interfaz 


import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import serial.tools.list_ports
import functools
import time
import serial


#Comprobacion de pto serie
ports = serial.tools.list_ports.comports()#chekeo que puertos detecta la pc ver que onda en Linux!
serialObj = serial.Serial() #creo instancia

def initComPort(index):
    currentPort=str(ports[index])
    print(currentPort)
    comPortVar= str(currentPort.split(' ')[0])
    serialObj.port= comPortVar
    serialObj.baudrate= 9600
    serialObj.open()

#Definir comunicacion con el aparatosky

#Boton help
#Cliquear para abrir -> Puerto X abierto
 
#Boton iniciar registro de temperaturas
#Boton detener registro
#Boton Guardar archivo
#  

###########################################################################################
                    #               AÑO-MES-DIA                 #    Grafico  T vs t_rel 
#   COM1   ON/OFF   #   (lo siguiente aparece al abrir el pto)  #
#                   #   hh:mm:ss    t_rel (s)    Temp (ºC)      #    Este debe tener eje t         
#                   #  10:00:00      00:00         20.3         #    actualizandose ctemente
#   Init_reg        #   este frame/canva va a ir agregando      #
#                   #   data constantemente, y debe ir mos      #
#   Det_reg         #   trando los ultimos en pantalla          # 
#                   #                                           #
#    Info           #                                           #
#                   #                                           #
#    Guardar        #                                           #               
###########################################################################################
#Con On/OFF podria haber un cambio de color Verde/Rojo
#Info: write(b'h\r')

root = tk.Tk()
#root.geometry('300x200')
#root.resizable(False, False)
root.title('Temperatura - Nombre_del_sensor')
root.config(bg='grey')

for onePort in ports:
    comBotton= Button(root,text=onePort,font=('Calibri','13'),height=1,width=40,command= functools.partial(initComPort,index=ports.index(onePort)) )
    comBotton.grid(row=ports.index(onePort),column=0)



while True:
    #LisSerialPorts() #Para armar la lista de botones (en caso de tener mas de 1 canal)
    #openSerialPort() #para abrir el puerto -  ColorOFF -> ColorON
    #printLiveTemp()  #comienza a mostrar valores y a hacer grafico
    #InitLog()        #empieza a grabar valores para exportar en .txt
    #StopLog()
    #SaveLog()        #abre ventana para elegir directorio/nombre para archivo    
    #
    
    root.update()     #actualiza la ventana  
    dataCanvas.config(scrollregion=dataCanvas.bbox('all'))#amplia el canva
#root.mainloop() 