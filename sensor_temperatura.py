#%%sensor_temperatura.py

from socket import timeout
from tkinter import *
import serial.tools.list_ports
import functools

ports = serial.tools.list_ports.comports()
for p in ports:
    print(p.name)
#serialObj = serial.Serial(port='COM3',baudrate=9600,timeout=1)


#%%
import time 
import serial 
import serial.tools.miniterm

serialObj = serial.Serial('COM4', baudrate=9600, stopbits=1,timeout=1) 
print('Port Details ->',serialObj)

print('readable:',serialObj.readable())
print('writable: ',serialObj.writable())
print('in_waiting: ',serialObj.in_waiting)
print('baudrate: ',serialObj.baudrate)
#%%
sendMsj = serialObj.write(b'b E')

print('BytesWritten = ', sendMsj)
ReceivedString = serialObj.read() #readline reads a string terminated by \n
print('Respuesta: ',ReceivedString.decode('utf8'))
#%%
EchoedVar    = serialObj.read() 
print (EchoedVar)
serialObj.close()
#%%
ReceivedString = serialObj.readline() #readline reads a string terminated by \n
print(ReceivedString)
#%%

def checkSerialPort():
    print(serialObj.isOpen(),serialObj.in_waiting)
    if serialObj.isOpen() and serialObj.in_waiting:
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf').rstrip('\n')
        print(recentPacketString)
        #Label(dataFrame,text=recentPacketString,font=('Calibri','13'),bg='white').pack() 
while True:
    #root.update()
    checkSerialPort()




#%% Receptor

ser2=serial.Serial('COM8',9600,serial.EIGHTBITS,timeout=1) 
if ser2.isOpen:
    print('ser2 open')
#%%
ser.write('todo bien'.encode('utf-8'))
ser2.readline().decode('utf-8')

#%%
print(ser.cts)
ser.close()
ser2.close()


#%%    
    
    recentPacket = ser.readlines()
    print([e.decode('utf') for e in recentPacket])
    for  e,i in enumerate(recentPacket):
        recentPacketString = e.decode('utf-8')
        print('recent' ,i, recentPacketString)
    
#%%
serialObj = serial.Serial('COM3', 9600, serial.EIGHTBITS,timeout=1)
#%%

print('open:',serialObj.is_open)
print('readable:',serialObj.readable())
print('in_waiting: ',serialObj.in_waiting)

#%%
root = Tk()
root.config(bg='grey')

serialObj.close()
currentPort=str(ports[0])
print(currentPort)
comPortVar= str(currentPort.split(' ')[0])
serialObj.port= comPortVar
serialObj.baudrate= 9600
serialObj.open()
#%%
#def initComPort(index):
    #serialObj.close()
    #currentPort=str(ports[index])
    #print(currentPort)
    #comPortVar= str(currentPort.split(' ')[0])
    #serialObj.port= comPortVar
    #serialObj.baudrate= 9600
    #serialObj.open()

for onePort in ports:
    comBotton = Button(root,text=onePort,font=('Calibri','13'),height=1,width=45,command= functools.partial(initComPort,index=ports.index(onePort)) )
    comBotton.grid(row=ports.index(onePort),column=0)

dataCanvas = Canvas(root,width=600,height=400,bg='white')
dataCanvas.grid(row=0,column=1,rowspan=100)

vsb = Scrollbar(root,orient='vertical',command= dataCanvas.yview)
vsb.grid(row=0,column=2,rowspan=100,sticky='ns')

dataCanvas.config(yscrollcommand=vsb.set)

dataFrame= Frame(dataCanvas,bg='white')
dataCanvas.create_window((10,0),window=dataFrame,anchor='nw')

#%%
def checkSerialPort():
    print(serialObj.isOpen(),serialObj.in_waiting)
    if serialObj.isOpen() and serialObj.in_waiting:
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf').rstrip('\n')
        print(recentPacketString)
        #Label(dataFrame,text=recentPacketString,font=('Calibri','13'),bg='white').pack() 
#%%

while True:
    #root.update()
    checkSerialPort()
    #dataCanvas.config(scrollregion=dataCanvas.bbox('all'))

#%%