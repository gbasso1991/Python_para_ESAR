#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:46:21 2023

@author: giuliano
"""

import os
import fnmatch
import time
from datetime import datetime,timedelta
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.optimize import curve_fit

#%% Levanto archivos del directorio
fname='Puntas_Hall_Corregido_3col.txt'

fpath=os.path.join(os.getcwd(),fname)
#%%
def Al(x,m,n):
    return x*m+n

def lector_archivos(path):
    '''Toma archivos .txt con datos en columna:
        B (Gauss)| Sensor 1 T (mV) | Sensor 0.1 T (mV)| 
    '''
    data = pd.read_table(path,header=4,
                                names=('B','V1','V2'),usecols=(0,1,2),
                                decimal='.',engine='python') 
    campo = pd.Series(data['B']).to_numpy(dtype=float)
    sonda1=pd.Series(data['V1']).to_numpy(dtype=float)
    sonda2=pd.Series(data['V2']).to_numpy(dtype=float)
    
    return campo, sonda1, sonda2
#%%
B, V1,V2=lector_archivos(fpath)

B=B/10000       # Gauss -> Tesla
V1= V1/1000     # mV -> V
V2= V2/1000     # mV -> V

(m1,n1),((err_m1,_),(_,err_n1))=curve_fit(Al, B, V1)
(m2,n2),((err_m2,_),(_,err_n2))=curve_fit(Al, B, V2)

print('-'*50)
print('Ajustes lineales:\n')
print(f'Sensor 1 (1 V/T):\nm1 = {m1} +/- {err_m1} V/T\n')
print(f'Sensor 2 (10 V/T):\nm2 = {m2} +/- {err_m2} V/T')
print('-'*50)

B_aux= np.linspace(0,1,1000)


fig, ax = plt.subplots(figsize=(7,4.5),constrained_layout=True)
ax.plot(B,V1,'o',label='Sensor 1V = 1T')
ax.plot(B_aux,Al(B_aux,m1,n1),label=f'AL Sensor 1T\nm$_1$ = {m1:.2f} V/T')
ax.plot(B,V2,'s',label='Sensor 1V = 0,1T')
ax.plot(B_aux,Al(B_aux,m2,n2),label=f'AL Sensor 0,1T\nm$_2$ = {m2:.2f} V/T')

plt.legend(ncol=2)
plt.grid()
plt.xlabel('Campo (T)')
plt.ylabel('Sonda Hall (V)')
plt.title('Sondas Hall vs. Campo VSM')
plt.show()
#%%

C_sonda1_V_to_T=1/m1
C_sonda2_V_to_T=1/m2

