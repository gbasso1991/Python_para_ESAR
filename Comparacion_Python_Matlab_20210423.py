#%%
# -*- coding: utf-8 -*-
"""
Script para comparar los archivos exportados de Planet_caravan_20210419.py y primavera_owon_20210408.m

@author: Giuliano
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from IPython import get_ipython 
import scipy

#1ero el ciclo importado de Python (modificar el path)
ruta = 'C:/Users/Giuliano/Documents/Doctorado en Fisica/Datos ESAR/111kHz_15A_100Mss_sc_Ciclo_de_histéresis.dat'
df = pd.read_table(ruta,sep='\s+',engine='python',decimal='.')
t = df.loc[:,'Tiempo_(s)']
H = df.loc[:,'Campo_(A/m)']
M = df.loc[:,'Magnetizacion_(A/m)']
#%%
#Ahora el ciclo importado de Matlab
ruta2 = 'C:/Users/Giuliano/Documents/Doctorado en Fisica/Datos ESAR/Datos Pedro/111kHz_15A_100Mss_sc_ciclo.dat'
df2 = pd.read_table(ruta2,sep='\s+',header=None,engine='python',decimal='.')
t2 = df2.loc[:,0] 
H2 = df2.loc[:,1]
M2 = df2.loc[:,2]


#%%
plt.figure()
plt.plot(H,M,'.-',label='Ciclo Python')
plt.plot(H2,M2,'-',label='Ciclo Matlab')
plt.legend(loc='best')
plt.title('Comparación de ciclos')
plt.xlabel('$H$ (A/m)')
plt.ylabel('$M$ (A/m)')
plt.grid()

plt.show()

#%%
plt.figure()
plt.subplot(211)
plt.plot(t,H,'d-',lw=0.2,label='Campo Python')
plt.plot(t2,H2,'.-',lw=0.2,label='Campo Matlab') 
#plt.plot(t,(H-H2)*500,'-',label='Resta Campo P-M (x500)')

plt.ylabel('$H$ (A/m)')

plt.legend(loc='best')
plt.grid()
plt.xlim(2e-6,3e-6)
plt.title('Comparación Campo y Magnetizacion para ciclos Python y Matlab')

plt.subplot(212)
plt.plot(t,M,'d-',lw=0.2,label='Magnetización Python')
plt.plot(t2,M2,'.-',lw=0.2,label='Magnetización Matlab')
#plt.plot(t,(M-M2)*50,'-',label='Resta Magnetizacion P-M (x50)')
plt.xlabel('t (s)')
plt.ylabel('$M$ (A/m)')
#plt.legend(loc='best')
plt.xlim(2e-6,3e-6)
plt.grid()
plt.show()

#%%
plt.figure()
plt.subplot(211)
plt.plot(t,H-H2,'-',label='Resta Campo P-M')
plt.legend(loc='best')
plt.grid()
plt.title('Resta de las señales' )

plt.subplot(212)
plt.plot(t,M-M2,'-',label='Resta Magnetizacion P-M')
plt.legend(loc='best')
plt.grid()
#%%
''' Fourier sobre las señales H(t) y M(t)'''

from scipy.fft import rfft, rfftfreq
t_f = np.array(t) 
campo =  np.array(H)
magnetizacion = np.array(M) 
duracion = t_f[-1] #seg
frec_muestreo_1 = 100*1e6 #muestras/seg

N = int(frec_muestreo_1*duracion)
yf = np.array(rfft(campo))
yf2= np.array(rfft(magnetizacion))
xf = np.array(rfftfreq(N , 1/frec_muestreo_1))
#%% Salvar incongruencias en los largos, revisar como hacerlo mejro
if len(xf) > len(yf):
    xf = np.resize(xf,len(yf))
elif len(xf) < len(yf):
    yf = np.resize(yf,len(xf))

if len(xf) > len(yf2):
    xf = np.resize(xf,len(yf2))
elif len(xf) < len(yf2):
    yf2 = np.resize(yf2,len(xf))

#%%
plt.figure()
plt.plot(xf/1000,abs(yf)/max(abs(yf)),'.-', label='Campo')
#plt.plot(xf/1000,abs(yf2)/max(abs(yf2)),'.-',label='Magnetizacion')
plt.xlim(0,1e3)
plt.grid()
#plt.vlines((xf[1]/1000,xf[3]/1000,xf[5]/1000),0,([yf2[1],yf2[3],yf2[5]]))
plt.legend(loc='best')

plt.xlabel('Frecuencia (kHz)')
plt.show()
#%% Obtengo los armonicos
armonicos_campo = [xf[1]] 
amp_arm_campo = [abs(yf[1]/max(abs(yf)))]
armonicos_mag = [xf[1],xf[3],xf[5]] #en Hz
amplitud_armonicos_mag = [abs(yf2[1])/max(abs(yf2)),abs(yf2[3])/max(abs(yf2)),abs(yf2[5])/max(abs(yf2))] #Normalizados

#%%

plt.plot(xf,abs(yf2)/max(abs(yf2)),'.-',label='$F_{\{M \}}$')
for i in range(0,len(armonicos)):
    plt.scatter(armonicos[i],amplitud_armonicos[i],c='r',label='%.2f kHz' %((armonicos[i])))
plt.xlim(0,1e6)
plt.vlines(armonicos, 0, amplitud_armonicos, color='r')
plt.legend(loc='best')
plt.xlabel('Frecuencia (Hz)')
plt.grid()

# %% Armo la sinusoide y comparo

g = amp_arm_campo[0]*np.sin(t_f*armonicos_campo[0])




plt.plot(t_f,g)
#plt.plot(t_f, campo/max(campo),label='H')
#plt.plot(t_f,magnetizacion/max(magnetizacion),label='M')
plt.legend()