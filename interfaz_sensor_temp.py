#%% Interfaz_sensor_temp.py
# Giuliano Basso 
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.pyplot import fill
import serial.tools.list_ports
import functools
import time
import serial
#%%

#Comprobacion de pto serie
# ports = serial.tools.list_ports.comports()#chekeo que puertos detecta la pc ver que onda en Linux!
# serialObj = serial.Serial() #creo instancia

# def initComPort(index):
#     currentPort=str(ports[index])
#     print(currentPort)
#     comPortVar= str(currentPort.split(' ')[0])
#     serialObj.port= comPortVar
#     serialObj.baudrate= 9600
#     serialObj.open()

#Definir comunicacion con el aparatosky

#Boton help
#Cliquear para abrir -> Puerto X abierto

#Boton iniciar registro de temperaturas
#Boton detener registro
#Boton Guardar archivo


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

# root = tk.Tk()
# #root.geometry('300x200')
# #root.resizable(False, False)
# root.title('Temperatura - Nombre_del_sensor')
# root.config(bg='grey')

# for onePort in ports:
#     comBotton= ttk.Button(root,text=onePort,font=('Calibri','13'),height=1,width=40,command= functools.partial(initComPort,index=ports.index(onePort)) )
#     comBotton.grid(row=ports.index(onePort),column=0)



# while True:
#     #LisSerialPorts() #Para armar la lista de botones (en caso de tener mas de 1 canal)
#     #openSerialPort() #para abrir el puerto -  ColorOFF -> ColorON
#     #printLiveTemp()  #comienza a mostrar valores y a hacer grafico
#     #InitLog()        #empieza a grabar valores para exportar en .txt
#     #StopLog()
#     #SaveLog()        #abre ventana para elegir directorio/nombre para archivo    
#     #

#     root.update()     #actualiza la ventana  
#     ttk.dataCanvas.config(scrollregion=ttk.dataCanvas.bbox('all'))#amplia el canva
# #root.mainloop() 
#%%



def abrir_puerto():
    print('Abrir puerto serie NAME')
    status_label.config(text='Puerto abierto',bg='green',fg='white')
    log_button['state']='normal'
    logoff_button['state']='normal'
    savelog_button['state']='normal'
    
    data_canvas = tk.Canvas(root,bg='white')
    data_canvas.grid(row=1,rowspan=4,column=2)

    graph_canvas = tk.Canvas(root,bg='white')
    graph_canvas.grid(row=1,rowspan=4,column=3)

    # Abrir el canva con datos y con grafico


def init_log():
    print('Comienza registro de temperatura')

def off_log():
    print('Donde guardar el registro de temperatura?')
    print('Abrir ventana de save')


def save_log():
    print('Detenido el registro de temperatura')

root = tk.Tk()
root.title('Sensor de Temperatura')
root.geometry("1200x400")
root.resizable(1,1)

#Grid config
root.columnconfigure(0,weight=1)
root.columnconfigure(1,weight=1)
root.columnconfigure(2,weight=3)
root.columnconfigure(3,weight=1)
#root.columnconfigure(4,weight=1)
#root.columnconfigure(5,weight=1)

#boton Puerto
port_button = ttk.Button(root,text='COM1',command=abrir_puerto)
port_button.grid(column=0,row=0,sticky=tk.E)
separador = ttk.Separator(root,orient='horizontal')
separador.grid(columnspan=2,sticky='WE')
#Label status puerto
status_label = tk.Label(root,text='Puerto cerrado',bg='grey',fg='black')
status_label.grid(column=1,row=0,sticky=tk.E)

#boton iniciar registro 
log_button = ttk.Button(root,text='Iniciar registro',command=init_log)
log_button.grid(row=2,column=0,columnspan=2,sticky='WE')
log_button['state']='disabled'


#boton detener registro 
logoff_button = ttk.Button(root,text='Detener registro',command=off_log)
logoff_button.grid(row=3,column=0,columnspan=2,sticky='WE')
logoff_button['state']='disabled'

#boton guardar
savelog_button = ttk.Button(root,text='Guardar registro',command=save_log)
savelog_button.grid(row=4,rowspan=2,column=0,columnspan=2,sticky=('WE','NS'))
savelog_button['state']='disabled'




root.mainloop()
# %%
