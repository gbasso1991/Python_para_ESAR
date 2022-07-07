#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Jun 23 13:31:35 2022
para levantar archivos de TG
@author: giuliano
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
import tkinter as tk
from tkinter import filedialog

#%%
root = tk.Tk()
root.withdraw()
texto_encabezado = "Seleccionar archivo a analizar: " 
path = filedialog.askopenfilenames(title=texto_encabezado,filetypes=(("Archivos .txt","*.txt"),("Archivos .dat","*.dat"),("Todos los archivos","*.*")))
directorio = path[0].rsplit('/',maxsplit=1)[0]
fnames_m = []

for item in path:    
    fnames_m.append(item.split('/')[-1])

#%% metadata 

with open(path[0], 'r') as file:
    for line in file:
        if 'Sample Name' in line:
            sample_name = line.split()[-1]
            print(f'Nombre de la muestra: {sample_name}')
        elif 'Sample Weight' in line:
            masa=float(line.split()[2]) 
            unidad = line.split()[3]
            print(f'masa {masa} {unidad}')
                      
file = np.loadtxt(path[0],skiprows=24,dtype=float)
tiempo = file[:,0 ] #seg
temperatura = file[:,1] #Celsius
peso = file[:,2] #mg
peso_porc = 100*peso/max(peso)
deriv_peso = np.gradient(peso_porc)



T_min= temperatura[(peso_porc==min(peso_porc)).nonzero()][0]
peso_min= peso_porc[(peso_porc==min(peso_porc)).nonzero()][0]


T_max= temperatura[(peso_porc==max(peso_porc)).nonzero()][0]
peso_max= peso_porc[(peso_porc==max(peso_porc)).nonzero()][0]


perdida_porc = peso_max-peso_min
perdida_masa = masa*perdida_porc/100
print(f'Weight Loss: {perdida_masa:.2f} {unidad} == {perdida_porc:.2f} %')

#%% Grafico
fig,ax =plt.subplots()
ax.plot(temperatura,peso_porc,'-')
#ax.plot(temperatura[(deriv_peso==0).nonzero()],peso_porc[(deriv_peso==0).nonzero()],'r.')
#ax.plot(temperatura[(peso_porc==max(peso_porc)).nonzero()],peso_porc[(peso_porc==max(peso_porc)).nonzero()],'b.')
#ax.plot(temperatura[(peso_porc==min(peso_porc)).nonzero()],peso_porc[(peso_porc==min(peso_porc)).nonzero()],'g.')

#ax2.plot(temperatura,deriv_peso,'-', alpha=0.7)

#plt.vlines(T_max, 50, peso_max,'r',lw=0.7)
#plt.hlines(peso_max,T_max,1,'r',lw=0.7)
plt.vlines(T_min, 50, peso_min,'r',lw=0.7)
plt.hlines(peso_min, 0, T_min,'r',lw=0.7, label=f'Perdida: {perdida_masa:.2f} mg = {perdida_porc:.2f} %')    
plt.legend()
plt.title(fnames_m[0] + ' - '+ sample_name) 
ax2=ax.twinx()
ax2.plot(temperatura,deriv_peso,'-',color='tab:orange', alpha=0.7)

ax.set_ylabel('Peso (%)')
ax2.set_ylabel('Deriv Peso (%/°C)')

ax.grid()
ax.set_xlabel('Temperatura (°C)')
#plt.xlim(0,temperatura[200])
plt.tight_layout()
plt.show()