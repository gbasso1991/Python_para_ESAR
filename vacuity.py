#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vacuity.py
Created on Tue Aug 30 2022

@author: Giuliano

Para levantar los archivos .csv de temperatura gnersados por el sensor Rugged
IMPORTANTE: debido a temas de sistema operativo los archivos recuperados de 
la pc ESAR tienen 3 hs de defasaje. 

ATENCION: Getting file creation dates, on the other hand, is fiddly 
          and platform-dependent: 
    On Windows, a file's ctime  stores its creation date. 
    You can access this in Python through os.path.getctime() or 
    the .st_ctime attribute of the result of a call to os.stat(). 
    This won't work on Unix, where the ctime is the last time that 
    the file's attributes or content were changed.

    On Linux, this is currently impossible, at least without writing 
    a C extension for Python. Although some file systems commonly used 
    with Linux do store creation dates, the Linux kernel offers no way 
    of accessing them.

"""
from re import M
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import os
import fnmatch
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from scipy.interpolate import BSpline

#% Seleccion Archivos muestra
root = tk.Tk()
root.withdraw()
texto_encabezado = "Seleccionar archivos con las medidas de muestra:"
path_m=filedialog.askopenfilenames(title=texto_encabezado,filetypes=(("Archivos .txt","*.txt"),("Archivos .dat","*.dat"),("Todos los archivos","*.*")))
directorio = path_m[0].rsplit('/',maxsplit=1)[0]

fnames_m = []
for item in path_m:    
    fnames_m.append(item.split('/')[-1])

print('Directorio de trabajo: '+ directorio +'\n')
print('Archivos de muestra: ')
for item in fnames_m:
    print(' >',item)
#%%
'''
Parámetros de la medida a partir de nombre del archivo de muestra: 'xxxkHz_yyydA_zzzMss_*.txt
'''
frec_nombre=[]      #Frec del nombre del archivo. Luego comparo con frec ajustada
Idc = []            #Internal direct current en el generador de RF
samples_per_s = []  #Base temporal 
for file in fnames_m:
    frec_nombre.append(float(file.split('_')[0][:-3])*1000)
    Idc.append(float(file.split('_')[1][:-2])/10)
    samples_per_s.append(1e-6/float(file.split('_')[2][:-3]))

'''
Los Horarios de las medidad difieren segun la maquina en la que 
se tengan los archivos. 
La fecha posta esta en mi cuaderno 
                            Paramagneto: 10:40:30 (pm01) a 10:43:26 (pm09_fond)
                            Agua: 11:07:30 (agua01) a 11:10:44 (agua09_fondo)
En la Notebook (date modified): 
                            Paramagneto: 06:40:30 (pm01) a 06:43:26 (pm09_fond)
                            Agua: 07:07:30 (agua01) a 07:10:44 (agua09_fondo)
                            
                            Son 04:00:00 hs de diferencia
'''
fecha_m =[] #fecha del archivo de la medida, OJO que es 'modification date' 

for path in path_m:
    fecha_m.append(datetime.fromtimestamp(os.path.getmtime(path)))

delta_t_m = [] #seg
for elem in fecha_m:
    delta_t_m.append(int((elem - fecha_m[0]).total_seconds()))


#%Ahora levanto el log de temperaturas en .csv

if fnmatch.filter(os.listdir(directorio),'*templog*'):
    dir_templog = os.path.join(directorio,fnmatch.filter(os.listdir(directorio),'*templog*')[0])

data = pd.read_csv(dir_templog,sep=';',header=5,
                    names=('Timestamp','Temperatura'),usecols=(0,1),
                    decimal=',',engine='python') 
temperatura = pd.Series(data['Temperatura']).to_numpy(dtype=float)

timestamp=[]
for time in pd.Series(data['Timestamp']):
    timestamp.append(time[11:19])
timestamp=np.array(timestamp,dtype=str)

#% Datetime de las medidas de muestra en funcion del horario del 1er dato

#primer dato: pm00 (a RT)
date_primer_dato = datetime(year=2022,month=8,day=30,hour=10,minute=40,second=14) #queda automatizar esto 
#para agua, cambiar por 11:06:00 
#date_primer_dato = datetime(year=2022,month=8,day=30,hour=11,minute=6,second=00) 

time_m = [] 
for elem in delta_t_m:
    time_m.append((date_primer_dato + timedelta(0,elem)).strftime('%H:%M:%S'))
time_m = np.array(time_m) #H:M:S del registro de cada archivo muestra

#obtengo los indices de estos horarios en el timestamp

temp_m = []
indx_temp = []
for t in time_m:
    indx_temp.append(np.flatnonzero(timestamp==t)[0])
    temp_m.append(temperatura[np.flatnonzero(timestamp==t)[0]])
temp_m=np.array(temp_m)
#% Printeo lo obtenido
print('Archivos de muestra: ')
for i, item in enumerate(fnames_m):
    print(item.rsplit('_')[-1][:-4],' ',time_m[i],' ',str(temp_m[i]) + ' ºC')

#%% dif temporal en s entre el comienzo del registro y la primer medida de muestra
#delta_0 = (datetime.strptime(time_m[0],'%H:%M:%S') - datetime.strptime(timestamp[0],'%H:%M:%S')).total_seconds()

#%%
# timestamp = pd.to_datetime(pd.Series(data['Timestamp']).to_numpy())
# aux=[]
# for element in timestamp:
#     aux.append(element.strftime('%H:%M:%S'))

#for e in timestamp:
 #   e = datetime.strptime(e[11:19],'%H:%M:%S')
    #print((e,)
#plt.xticks(rotation=45, ha='right')
#%% Ploteo los ciclos y asocio la temperatura
#traigo los datos desde possessor
#%%
Ciclos_eje_H=Ciclos_eje_H
Ciclos_eje_M=Ciclos_eje_M
Ciclos_eje_H_cal=Ciclos_eje_H_cal
Ciclos_eje_M_cal_ua=Ciclos_eje_M_cal_ua
SAR=SAR

fig = plt.figure(figsize=(10,8),constrained_layout=True)
ax = fig.add_subplot(1,1,1)
axin = ax.inset_axes([0.60,0.08, 0.35,0.38])
axin.set_title('Calibración',loc='center')
#axin.yaxis.tick_right()
plt.setp(axin.get_yticklabels(),visible=True)
plt.setp(axin.get_xticklabels(),visible=True)
axin.yaxis.tick_right()
axin.grid()
axin.axhline(0,0,1,lw=0.9,c='k')
axin.axvline(0,0,1,lw=0.9,c='k')

for i in range(len(fnames_m)):      
    plt.plot(Ciclos_eje_H[i],Ciclos_eje_M[i],label=f'{fnames_m[i][-8:-4]}   {temp_m[i]}ºC')
    axin.plot(Ciclos_eje_H_cal[i], Ciclos_eje_M_cal_ua[i])
    axin.set_ylabel('M $(V\cdot s)$')
    axin.set_xlabel('H $(A/m)$')

plt.legend(loc='best',fancybox=True)
plt.grid()
plt.xlabel('Campo (A/m)',fontsize=15)
plt.ylabel('Magnetización (A/m)',fontsize=15)
plt.suptitle('Ciclos de histéresis en descongelamiento',fontsize=30)
plt.show()
#plt.savefig('Ciclos_histeresis_descong.png',dpi=300,facecolor='w')

# %%
'''Realizo ajustes lineales sobre los ciclos, obtengo pendientes m
Grafico m(t), m(T)'''



pendiente_filtrada , _ = np.polyfit(campo_c,magnetizacion_ua_c,1) #[pendiente]=m*V*s/A  [ordenada]=V*s
#%%
from sklearn.linear_model import LinearRegression
from scipy import stats

pendiente_AL = []
pendiente_AL = []
pendiente_AL_err = []
ordenada_AL = []
for i in range(len(fnames_m)):
    slope,intercept,r_value,p_value,std_err = stats.linregress(Ciclos_eje_H[i],Ciclos_eje_M[i])
    pendiente_AL.append(slope)
    pendiente_AL_err.append(std_err)
    ordenada_AL.append(intercept)    
    #x = Ciclos_eje_H[i].reshape((-1, 1))
    #y = Ciclos_eje_M[i].reshape((-1, 1))
    #model = LinearRegression(fit_intercept=True).fit(x,y)
    #r_sq = model.score(x, y)
    print(fnames_m[i])
    print(f"Ordenada: {intercept}")
    print(f"Pendiente: {slope}")
    print(f"R^2: {r_value}\n")

    #y_pred=model.predict(x)
    #y_pred_bis = x*slope+intercept


#%%
#pendiente_AL.std()

#plt.plot(delta_t_m,pendiente_AL,'o-',label=directorio.rsplit('/')[-1])
plt.errorbar(delta_t_m,pendiente_AL,xerr=1,yerr=pendiente_AL_err,lw=0.8,
                    capsize=1,label=directorio.rsplit('/')[-1])

plt.xlabel('$\Delta t$ (s)')
plt.ylabel('pendiente M/H')
plt.grid()
plt.title('Pendiente en funcion de t')
plt.legend()
plt.show()

plt.errorbar(temp_m,pendiente_AL,yerr=pendiente_AL_err,lw=0.8,
                    capsize=1,label=directorio.rsplit('/')[-1])
plt.xlabel('T (ºC)')
plt.ylabel('pendiente M/H')
plt.grid()
plt.title('Pendiente en funcion de T')
plt.legend()
plt.show()
# %%

fig = plt.figure(figsize=(10,8),constrained_layout=True)
ax = fig.add_subplot(1,1,1)

for i in range(len(fnames_m)):     
    H_aux = np.linspace(min(Ciclos_eje_H[i]),max(Ciclos_eje_H[i]),2000)
    M_aux = H_aux*pendiente_AL[i]+ordenada_AL[i]
    plt.plot(H_aux,M_aux,label=f'{fnames_m[i][-8:-4]}\n{temp_m[i]}ºC\n{time_m[i]}')

    
axin = ax.inset_axes([0.60,0.09, 0.4,0.38])
axin.set_title('m($\Delta$t)',loc='center')
axin.grid()
axin.errorbar(delta_t_m,pendiente_AL,xerr=1,yerr=pendiente_AL_err,lw=0.8,
                    capsize=1,label=directorio.rsplit('/')[-1])
axin.legend()
axin.set_ylabel('pendiente')
axin.set_xlabel('$\Delta$t')
axin2 = ax.inset_axes([0.09,0.55, 0.4,0.38])
axin2.set_title('m(T)',loc='center')

plt.setp(axin2.get_yticklabels(),visible=True)
plt.setp(axin2.get_xticklabels(),visible=True)

axin2.grid()
axin2.errorbar(temp_m,pendiente_AL,yerr=pendiente_AL_err,lw=0.8,
                    capsize=1,label=directorio.rsplit('/')[-1])
axin2.set_ylabel('pendiente')
axin2.set_xlabel('T (ºC)')
axin2.legend()
plt.legend(fancybox=True,ncol=5,bbox_to_anchor=(0.5,-0.11),loc='upper center')

plt.grid()
plt.xlabel('Campo (A/m)',fontsize=15)
plt.ylabel('Magnetización (A/m)',fontsize=15)
plt.suptitle('Ciclos de histéresis en descongelamiento',fontsize=30)
plt.savefig('pendientes_en_tiempo_y_temperatura.png',faceolor='w',dpi=300)
# %%
