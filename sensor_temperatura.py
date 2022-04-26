#%%sensor_temperatura.py
from tkinter import *
import serial.tools.list_ports
import functools

ports = serial.tools.list_ports.comports()
serialObj = serial.Serial()


root = Tk()
root.config(bg='grey')

def initComPort(index):
    currentPort=str(ports[index])
    #print(currentPort)
    comPortVar= str(currentPort.split(' ')[0])
    serialObj.port= comPortVar
    serialObj.baudrate= 9600
    serialObj.open()

for onePort in ports:
    comBotton= Button(root,text=onePort,font=('Calibri','13'),height=1,width=45 )
    comBotton.grid(row=ports.index(onePort),column=0,command= functools.partial(initComPort,index=ports.index(onePort)))

dataCanvas = Canvas(root,width=600,height=400,bg='white')
dataCanvas.grid(row=0,column=1,rowspan=100)

vsb = Scrollbar(root,orient='vertical',command= dataCanvas.yview)
vsb.grid(row=0,column=2,rowspan=100,sticky='ns')

dataCanvas.config(yscrollcommand=vsb.set)

dataFrame= Frame(dataCanvas,bg='white')
dataCanvas.create_window((10,0),window=dataFrame,anchor='nw')


def checkSerialPort():
    if serialObj.isOpen() and serialObj.in_waiting:
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf').rstrip('\n')
        Label(dataFrame,text=recentPacketString,font=('Calibri','13'),bg='white').pack() 



while True:
    root.update()
    checkSerialPort()
    dataCanvas.config(scrollregion=dataCanvas.bbox('all'))

