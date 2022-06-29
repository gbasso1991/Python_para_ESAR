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

file = np.loadtxt(fnames_m[0],skiprows=16,dtype=float)
tiempo = file[:,0 ] #seg
temperatura = file[:,1] #Celsius
peso = file[:,2] #mg
peso_porc = 100*peso/max(peso)
deriv_peso = np.gradient(peso_porc)

#grafico
fig,ax =plt.subplots()
ax.plot(temperatura,peso_porc,'.-',zorder=3)
ax.set_ylabel('Peso (%)')
ax.grid()
ax.set_xlabel('Temperatura (°C)')

ax2=ax.twinx()
ax2.plot(temperatura,deriv_peso,'   -',c='tab:orange',zorder=4)
ax2.set_ylabel('Derv Peso (%/°C)')

plt.title(fnames_m[0][:-4])

plt.xlim(0,max(temperatura))

plt.show()