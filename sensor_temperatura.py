#%%sensor_temperatura.py

from msilib.schema import ServiceInstall
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

serialInst = serial.Serial()  # Instancia vacia del objeto Serial
#%%#para leer armo loop
portList =[]

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))


val = input('Seleccionar puerto: COM')
print(val)

for x in range(len(portList)):
    if portList[x].startswith('COM'+str(val)):
        portVar='COM'+str(val)
        print(portList[x])
#%%
serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open() 

while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        print(packet.decode('bit'))


# %%
import serial
ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM3'