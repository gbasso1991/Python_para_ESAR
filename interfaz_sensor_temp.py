#%% Interfaz_sensor_temp.py
# Giuliano Basso 
# https://realpython.com/python-encodings-guide/
# from tkinter import *
# import serial.tools.list_ports
# import functools
# import time

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

#

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
import serial
from tkinter.filedialog import asksaveasfile,askdirectory
import matplotlib.pyplot as plt
import numpy as np

aux_time = np.linspace(0,60,60)
aux_temp = np.linspace(25.0,30.0,60)
fig, ax = plt.subplots(figsize=(5,5))
    

def abrir_puerto():
    print('Abrir puerto serie NAME')
    status_label.config(text='Puerto abierto',bg='green',fg='white')
    log_button['state']='normal'
    closeport_button['state']='normal'

    # define columns
    columns = ('horario', 'tiempo', 'Temperatura')
    tree1 = ttk.Treeview(root, columns=columns, show='headings')
        
    # define headings
    tree1.heading('horario', text='hh:mm:ss')
    tree1.heading('tiempo', text='t (s)')
    tree1.heading('Temperatura', text='Temperatura (ºC)')

    # generate sample data
    aux_list = []
    for n in range(1, 100):
        aux_list.append((f'hh:mm:ss {n}', f'{n}', f'25.4'))

    # add data to the tree1view
    #for elem in aux_list:
        #tree1.insert('', tk.END, values=elem)

    # def item_selected(event):
    #     for selected_item in tree1.selection():
    #         item = tree1.item(selected_item)
    #         record = item['values']
    #         # show a message
    #         showinfo(title='Information', message=','.join(record))
    # tree1.bind('<<Tree1viewSelect>>', item_selected)

    tree1.grid(row=1, column=2,rowspan=6, sticky='nsew')

    # scrollbar
    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree1.yview)
    tree1.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=1, column=3, rowspan=7,sticky='ns')

def cerrar_puerto():
    print('Puerto NAME cerrado - reiniciar puerto? flush')
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

root = tk.Tk()
root.title('Sensor de Temperatura')
root.geometry("1200x400")
root.resizable(1,1)

#Grid config
root.columnconfigure(0,weight=1)
root.columnconfigure(1,weight=1)
root.columnconfigure(2,weight=1)
root.columnconfigure(3,weight=1)
#root.columnconfigure(4,weight=1)
#root.columnconfigure(5,weight=1)

#boton Puerto
port_button = ttk.Button(root,text='COM1',command=abrir_puerto)
port_button.grid(column=0,row=0,sticky='nsew')

closeport_button = ttk.Button(root,text='Cerrar COM1',command=cerrar_puerto)
closeport_button.grid(column=1,row=1,sticky='nsew')
closeport_button['state']='disbled'

# separador = ttk.Separator(root,orient='horizontal')
# separador.grid(columnspan=2,sticky='WE')

#Label status puerto
status_label = tk.Label(root,text='Puerto cerrado',bg='grey',fg='black')
status_label.grid(column=1,row=0,sticky='nsew')

date_label = tk.Label(root,text='AAAA-MM-DD hh:mm:ss',bg='grey',fg='white')
date_label.grid(column=2,row=0,stick='nsew')

#boton iniciar registro 
log_button = ttk.Button(root,text='Iniciar registro',command=init_log)
log_button.grid(row=2,column=0,columnspan=2,sticky='NSWE')
log_button['state']='disabled'

#boton pausar registro 
pauselog_button = ttk.Button(root,text='Pausar registro',command=pause_log)
pauselog_button.grid(row=3,column=0,columnspan=2,sticky='NSWE')
pauselog_button['state']='disabled'


#boton detener registro 
stoplog_button = ttk.Button(root,text='Detener registro',command=stop_log)
stoplog_button.grid(row=4,rowspan=2,column=0,columnspan=2,sticky='NSWE')
stoplog_button['state']='disabled'

#boton guardar
savelog_button = ttk.Button(root,text='Guardar registro',command=save_log)
savelog_button.grid(row=6,rowspan=2,column=0,columnspan=2,sticky='nsew')
savelog_button['state']='disabled'




root.mainloop()
# %%
