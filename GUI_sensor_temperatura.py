#%%sensor_temperatura.py
#https://realpython.com/python-encodings-guide/
from tkinter import *
import serial.tools.list_ports
import functools
import time

ports = serial.tools.list_ports.comports()

serialObj = serial.Serial()

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
# %%
