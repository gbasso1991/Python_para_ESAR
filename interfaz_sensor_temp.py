#%% Interfaz_sensor_temp.py
# Giuliano Basso 

# ports = serial.tools.list_ports.comports()#chekeo que puertos detecta la pc
# serialObj = serial.Serial() #creo instancia

# root = Tk() 
# root.config(bg='grey')

# def initComPort(index):
#     currentPort=str(ports[index])
#     print(currentPort)
#     comPortVar= str(currentPort.split(' ')[0])
#     serialObj.port= comPortVar
#     serialObj.baudrate= 9600
#     serialObj.open()

# for onePort in ports:
#     comBotton= Button(root,text=onePort,font=('Calibri','13'),height=1,width=40,command= functools.partial(initComPort,index=ports.index(onePort)) )
#     comBotton.grid(row=ports.index(onePort),column=0)

# dataCanvas = Canvas(root,width=600,height=400,bg='white')
# dataCanvas.grid(row=0,column=1,rowspan=100)

# vsb = Scrollbar(root,orient='vertical',command= dataCanvas.yview)#barra de desplazamiento vertical
# vsb.grid(row=0,column=2,rowspan=100,sticky='ns')

# dataCanvas.config(yscrollcommand=vsb.set)
# dataFrame= Frame(dataCanvas,bg='white')
# dataCanvas.create_window((10,0),window=dataFrame,anchor='nw')


# def checkSerialPort():
#     if serialObj.isOpen():
#         serialObj.write(b't1\r')
        
#         recentPacket = serialObj.readline()
#         recentPacketString = recentPacket.decode('utf-8','strict').rstrip('\n*')
        
#         Label(dataFrame,text=recentPacketString,font=('Calibri','13'),bg='white').pack() 
#         #time.sleep(0.1)
#     else:
#         pass

# while True:

#     checkSerialPort()
#     root.update()
#     dataCanvas.config(scrollregion=dataCanvas.bbox('all'))

#%% Esquema del mio

#Comprobacion de pto serie
# ports = serial.tools.list_ports.comports()#chekeo que puertos detecta la pc ver que onda en Linux!
# serialObj = serial.Serial() 
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

#Info: write(b'h\r')

#%%
import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter.scrolledtext import ScrolledText
from matplotlib.pyplot import fill
import serial.tools.list_ports
import functools
import time
from datetime import datetime, timedelta
import serial
from tkinter.filedialog import asksaveasfile,askdirectory
import matplotlib.pyplot as plt
import numpy as np
#%%

def display_time():
    '''
    Funcion para obtener la hora y actualizar cada 1 seg
    '''
    current_time = time.strftime(('%Y/%m/%d - %I:%M:%S %p'))
    date_label['text']=current_time
    root.after(1000,display_time)
    

def detectar_puertos():
    ''
    ports_detected = serial.tools.list_ports.grep(regexp='USB',include_links=True)#chekeo que puertos detecta la pc
    for port in ports_detected:
        currentPort=str(port)
        print(currentPort)
        comPortVar= str(currentPort.split(' ')[-1])
        print(comPortVar)
        print('---')

def getTemp(serialObj,t_0):
    '''Toma un objeto Serial y un tiempo inicial.
    Se comunica con el sensor y lee respuesta
    Devuelve tupla: (horario, tiempo abosoluto, Temperatura)'''
    
    if serialObj.isOpen():
        serialObj.write(b't1\r')
        t_aux =datetime.now()
        recentPacket = serialObj.readline()
        recentPacketString = recentPacket.decode('utf-8','ignore').rstrip('\n*')
        #print(recentPacketString)
        #Label(dataFrame,text=recentPacketString,bg='white').pack(expand=True) 
        #time.sleep(0.1)
        data_tuple =(t_aux.strftime('%H:%M:%S'), (t_aux-tiempo_0).seconds, recentPacketString)
        print(data_tuple)
        return data_tuple
    
def abrir_puerto(name):
    global tiempo_0
    tiempo_0 = datetime.now() 
    
    status_label.config(text='Puerto abierto',bg='green',fg='white')
    log_button['state']='normal'
    closeport_button['state']='normal'
    
    #Creo la instacia del pto serie
    global serialObj
    serialObj = serial.Serial(port=name,
                              baudrate=9600,
                              bytesize=8,
                              stopbits=1)
    if serialObj.isOpen():
        print(f'puerto serie {name} abierto')

    data_packet= getTemp(serialObj=serialObj,t_0=tiempo_0) #Se comumnica con el sensor recupera tupla con info
    
    #print de datos en pantalla
    # define columns
    columns = ('horario', 'tiempo', 'Temperatura')
    tree1 = ttk.Treeview(root, columns=columns, show='headings', height=10)
    #tv.column('#0', width=0, stretch=NO)
    tree1.column('horario', anchor='center', width=100)
    tree1.column('tiempo', anchor='center', width=100)
    tree1.column('Temperatura', anchor='center', width=100)    
    # define headings
    tree1.heading('horario', text='hh:mm:ss',anchor='center')
    tree1.heading('tiempo', text='t (s)',anchor='center')
    tree1.heading('Temperatura', text='Temperatura (ºC)',anchor='center')

    # add data to the tree1view
    #for elem in aux_list:
    tree1.insert('', tk.END, values=data_packet)


    tree1.grid(row=1, column=2,rowspan=10, sticky='nsew')

    # scrollbar
    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree1.yview)
    tree1.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=1, column=3, rowspan=10,sticky='ns')

    #para el grafico T vs. t
    dataCanvas = tk.Canvas(root,width=400,height=400,bg='white')
    dataCanvas.grid(row=0,column=3,rowspan=100,sticky='nsew')
    

def cerrar_puerto(serialObj):
    if serialObj.isOpen():
        serialObj.close()
        print(f'Puerto serie {serialObj.name} cerrado')
        status_label.config(text='Puerto cerrado',bg='red',fg='black')
        date_label.config(text='Fecha',bg='grey',fg='white')
        log_button['state']='disabled'
        stoplog_button['state']='disabled'

def init_log():
    print('Adquisicion iniciada')
    stoplog_button['state']='normal'
    pauselog_button['state']='normal'
    date_label.config(text='Fecha',bg='green',fg='white')
    #iniciar_funcion_que_interactua_con_sensor() 
  
def pause_log():
    #pausar_funcion_que_interactua_con_sensor()
    if pauselog_button['text']=='Pausar registro':
        pauselog_button['text']='Reanudar registro'
        date_label.config(text='Fecha',bg='orange',fg='white')
        print('Adquisicion pausada')
    else:
        pauselog_button['text']='Pausar registro'
        date_label.config(text='Fecha',bg='green',fg='white')
        print('Adquisicion reanudada')

def stop_log():
    print('Adquisicion detenida')
    #detener_funcion_que_interactua_con_sensor()
    savelog_button['state']='normal'  
    date_label.config(text='Fecha',bg='orange',fg='white')

def save_log():
    Files=[('.txt','*.txt')]
    archivo = asksaveasfile(mode='w', initialdir=os.getcwd(),filetypes=Files)
    print(f'Registro de temperatura guardado en SAVEDIR')
    

#%%
root = tk.Tk()
root.title('Sensor de Temperatura')
root.geometry("1200x400")
root.resizable(1,1)


#ports_detected = serial.tools.list_ports.grep(regexp='USB',include_links=True)
ports_detected = serial.tools.list_ports.comports() #chekeo que puertos detecta la pc
serialObj = serial.Serial() #creo instancia


#Grid config
root.columnconfigure(0,weight=1)
root.columnconfigure(1,weight=1)
root.columnconfigure(2,weight=1)
root.columnconfigure(3,weight=1)
#root.columnconfigure(4,weight=1)
#root.columnconfigure(5,weight=1)

#Label status puerto
status_label = tk.Label(root,text='Puerto cerrado',bg='grey',fg='black')
status_label.grid(column=1,row=0,sticky='nsew')

#Label fecha y hora
date_label = tk.Label(root,text='Tiempo',bg='grey',fg='white')
date_label.grid(column=2,row=0,stick='nsew')

#Boton iniciar registro 
log_button = ttk.Button(root,text='Iniciar registro',command=init_log)
log_button.grid(row=3,column=0,columnspan=2,sticky='NSWE')
log_button['state']='disabled'

#Boton pausar registro 
pauselog_button = ttk.Button(root,text='Pausar registro',command=pause_log)
pauselog_button.grid(row=4,column=0,columnspan=2,sticky='NSWE')
pauselog_button['state']='disabled'


#Boton detener registro 
stoplog_button = ttk.Button(root,text='Detener registro',command=stop_log)
stoplog_button.grid(row=5,rowspan=2,column=0,columnspan=2,sticky='NSWE')
stoplog_button['state']='disabled'

#Boton guardar
savelog_button = ttk.Button(root,text='Guardar registro',command=save_log)
savelog_button.grid(row=7,rowspan=2,column=0,columnspan=2,sticky='nsew')
savelog_button['state']='disabled'

#Boton Puerto
#port_button = ttk.Button(root,text='COM1',command=abrir_puerto)
#port_button.grid(column=0,row=0,sticky='nsew')

# separador = ttk.Separator(root,orient='horizontal')
# separador.grid(columnspan=2,sticky='WE')



for indice,puerto in enumerate(ports_detected):
    port_button= tk.Button(root,text=str(puerto[indice]),
                           height=1,width=40,
                           command=functools.partial(abrir_puerto,name=ports_detected[indice].device))
    port_button.grid(row=indice,column=0,sticky='nsew')

closeport_button = ttk.Button(root,
                              text='Cerrar puerto',
                              command=functools.partial(cerrar_puerto,serialObj=serialObj))

closeport_button.grid(column=1,row=1,sticky='nsew')
closeport_button['state']='disbled'
 
display_time()
root.mainloop()






##