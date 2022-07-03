#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculo de ciclos de sistemas que en equilibro se comportan como Langevin.

La ecuación a satisfacer es: 

d M(H,t) / dt = (1/tau) ( Meq(H) - M(H,t) )

M. Shliomis, Sov. Phys. Uspekhi (Engl. transl.) 17 (2) (1974) 153.


H(t) = H0 Sin(wt)
Meq(H(t)) = Ms Langevin (A H(t))
A = mu0 mub Nmb / (kB T) 

El argumento de la Langevin tambien se puede escribir:
A H(t) = A0 Sin(wt), con A0 = A H0

"""

import numpy as np
import matplotlib.pyplot as plt
from   scipy.integrate import odeint

def Langevin(x):
    """Función de Langevin"""
    return 1./np.tanh(x) -1./(x)

# Definimos la derivada que aparece en la ecuacion diferencial.
def dm_dt(m, t, tau, A0, w):
    aux = A0*np.sin(w*t) #argumento de la langevin
    meq = Langevin(aux)
    return (1./tau)*(meq-m)
    
#%%
#Parametros del Campo
H0 = 0.351*4500*10**3                  # Amplitud campo [A/m].  
frec = 100e3;                   # Frecuencia del campo [1/s].
N = 3                           # Número de ciclos.

#Parametros de la particula
tau = 400*1e-9                  # Tiempo de relajacion [s].
Nmb = 1000                       # Número de magnetones de Bhor por partícula.
Ms  = 1                         # Magnetizacion de saturación [A/m].

#Temperatura
T = 300                         # Temperatura [K].

#Constantes
pi  = np.pi
mu0 = 4*pi*1e-7                 # permeabilidad del vacio [Tm/A].
mub = 9.2740097e-24             # Magneton de Bhor [J/T].
kB  = 1.3806488e-23             # Constante de Boltzman [J/T].

#=======================

tiempo = N / frec * np.arange(0.5e-3,1,1e-3)
A      = mu0 * mub * Nmb / ( kB * T ) 
campo  = H0 * np.sin(2 * pi * frec * tiempo)
A0     = A*H0
w      = 2*pi*frec

#=======================
      
# Respuesta de equilibrio.
# M = Ms L(A H)
aux     = A * campo                 #Argumento de la Langevin.
magnet_eq = Ms * Langevin(aux);     #Magnetizacion de equilibrio

# Aproximación de bajo campo con susceptibilidad "Chord".
# M(H(t)) ~ Xi_chord *H0 Sin(w t - arctan(w tau))
# M(H0) = Ms L(A H0) ~ Xi_chord H0 => Xi_chord = Ms L(A H0) / H0
Xi_chord = Ms * Langevin(A*H0) / H0
magnet_bc_ch = Xi_chord * H0 * np.sin(w * tiempo - np.arctan(w * tau))

# Resolución  numérica.
m0    = 0.0        #Valor inicial de M.
m_ec  = odeint(dm_dt, m0, tiempo,args=(tau, A0, w))
magnet_ec_dif = Ms* np.array(m_ec).flatten()

#%%
#Gráficas  
plt.figure(figsize=(10,8),constrained_layout=True)

f1 = plt.subplot(2,1,1)
plt.plot(campo, magnet_eq,     'k-',label='Equilibrio')
plt.plot(campo, magnet_bc_ch,  'b-',label='Aproximación bajo campo')
plt.plot(campo, magnet_ec_dif, 'r-',label='Ecuacion diferencial')
plt.grid()
plt.legend(loc='best')
f1.set_ylabel('Magnetización/$M_s$')
f1.set_xlabel('Campo (A/M)')
f1.set_title('Frecuencia {:.0f} kHz, Campo {:.0f} kA/m'.format(frec/1000, H0/1000),fontsize=18)

f2 = plt.subplot(2,1,2)
plt.plot(tiempo, magnet_eq/max(magnet_eq),        'k-',label='Equilibrio')
plt.plot(tiempo, magnet_bc_ch/max(magnet_bc_ch),  'b-',label='Aproximación bajo campo')
plt.plot(tiempo, magnet_ec_dif/max(magnet_ec_dif),'r-',label='Ecuacion diferencial')
plt.plot(tiempo, campo/H0,                        'm-',label='Campo')
plt.grid()
plt.legend(loc='best',fontsize=9)
f2.set_ylabel('Magnetizacion y Campo\nNormalizados')
f2.set_xlabel('Tiempo (s)')


# %%
