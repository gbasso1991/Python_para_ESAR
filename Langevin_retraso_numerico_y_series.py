#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculo de ciclos de sistemas que en equilibro se comportan como Langevin.

La ecuación a satisfacer es: 

     d M(H,t) / dt = (1/tau) ( Meq(H) - M(H,t) )

M. Shliomis, Sov. Phys. Uspekhi (Engl. transl.) 17 (2) (1974) 153.

H(t)      = H0 Cos(wt)
Meq(H(t)) = Ms Langevin (A H(t))
A         = mu0 mub Nmb / (kB T) 

El argumento de la Langevin tambien se puede escribir:
A H(t) = A0 Cos(wt), con A0 = A H0

Se resuelve por dos maneras:
    -numericamente 
    -analiticamente utilizando series

"""

import numpy as np
import matplotlib.pyplot as plt
from   scipy.integrate import odeint
import math
import scipy

def Langevin(x):
    """Función de Langevin"""
    return 1./np.tanh(x) -1./x

# Definimos la derivada que aparece en la ecuacion diferencial.
def dm_dt(m, t, tau, A0, w):
    aux = A0*np.cos(w*t) #argumento de la langevin
    meq = Langevin(aux)
    return (1./tau)*(meq-m)
    
#Parametros del Campo
H0 = 0.6*0.351*4500*10**3       # Amplitud campo [A/m].  
frec = 100e3                    # Frecuencia del campo [1/s].
N = 3                           # Número de ciclos.

#Parametros de la particula
tau = 400*1e-9                  # Tiempo de relajacion [s].
Nmb = 1000                      # Número de magnetones de Bhor por partícula.
Ms  = 1                         # Magnetizacion de saturación [A/m].

#Temperatura
T = 300                         # Temperatura [K].

#Constantes
pi  = np.pi
mu0 = 4*pi*1e-7                 # permeabilidad del vacio [Tm/A].
mub = 9.2740097e-24             # Magneton de Bhor [J/T].
kB  = 1.3806488e-23             # Constante de Boltzman [J/T].

#=======================
# parametros auxiliares
tiempo = N / frec * np.arange(0.5e-3,1,1e-3)
A      = mu0 * mub * Nmb / ( kB * T ) 
campo  = H0 * np.cos(2 * pi * frec * tiempo)
A0     = A*H0
w      = 2*pi*frec
#===========================================
# Respuesta de equilibrio.
# M = Ms*L(A H)
aux       = A * campo               #Argumento de la Langevin.
magnet_eq = Ms * Langevin(aux);     #Magnetizacion de equilibrio
#===========================================

# Resolución  numerica.
m0    = 0.0        #Valor inicial de M.
m_ec  = odeint(dm_dt, m0, tiempo,args=(tau, A0, w))
magnet_ec_dif = Ms* np.array(m_ec).flatten()

#=========================================== 
# Respuesta de obtenidas en series de Laurent
# M = Ms L(A H)
     
# Cálculo de los números de Bernoulli según Oldham, pag 41.
Bjc = [1.]
for k in range(160):
    n = len(Bjc)
    nf = np.math.factorial(n)
    v = 0
    for j in range(n):
        jf = np.math.factorial(j)
        ff = np.math.factorial(n+1-j)
        v += Bjc[j]/jf/ff
    Bjc.append(-nf*v)

#Equilibrio
def Langevin_serie_time(t,ho,w,mmax,lmax):
    lang_approx_time = 0.
    for i in range(1,mmax):
        alfam=0
        for j in range(1,lmax):
            a1=2*(i+j-1)
            a3=2*i+2*j-3
            alfam+=4*Bjc[a1]/math.factorial(a1)*scipy.special.binom(a3,j-1)*ho**(a3)
        lang_approx_time+= alfam*np.cos((2*i-1)*w*t)
    return lang_approx_time

#Fuera de equilibrio
def Langevin_serie_time_retraso(t,ho,w,tau,mmax,lmax):
    lang_approx_time = 0.
    for i in range(1,mmax):
        alfam=0
        for j in range(1,lmax):
            a1=2*(i+j-1)
            a3=2*i+2*j-3
            corr=1/(1+((2*i-1)*w*tau)**2)**.5
            alfam+=corr*4*Bjc[a1]/math.factorial(a1)*scipy.special.binom(a3,j-1)*ho**(a3)
        lang_approx_time+= alfam*np.cos((2*i-1)*w*t-np.arctan(i*w*tau))
    return lang_approx_time
#===========================================
#%% 
# Respuesta en series de Laurent
#Equilibrio
magnet_eq_serie = Ms *Langevin_serie_time(tiempo,A*H0,2*pi*frec,20,60)
#Fuera de equilibrio
magnet_eq_serie_retraso = Ms *Langevin_serie_time_retraso(tiempo,A*H0,2*pi*frec,tau,20,50)

#Gráficas  

plt.figure(figsize=(6,10),constrained_layout=True)

f1 = plt.subplot(3,1,1)
plt.plot(campo, magnet_eq,'k-',label='Equilibrio')
#plt.plot(campo, magnet_bc_ch,'b-',label='Aproximación bajo campo')
plt.plot(campo, magnet_ec_dif,'r-',label='Ecuacion diferencial')
plt.plot(campo, magnet_eq_serie_retraso,'g-',label='Serie')

plt.legend(loc='best')
f1.set_ylabel('Magnetización/M_s')
f1.set_xlabel('Campo (A/M)')
f1.set_title('Frecuencia {:.0f} kHz, Campo {:.0f} kA/m'.format(frec/1000, H0/1000))

f2 = plt.subplot(3,1,2)
#plt.plot(tiempo, magnet_eq/max(magnet_eq),'k-',label='Equilibrio')
#plt.plot(tiempo, magnet_bc_ch/max(magnet_bc_ch),  'b-',label='Aproximación bajo campo')
#plt.plot(tiempo, magnet_ec_dif/max(magnet_ec_dif),'r-',label='Ecuacion diferencial')
#plt.plot(tiempo, magnet_eq_serie_retraso/max(magnet_eq_serie_retraso),'g-',label='Equilibrio Serie')
#plt.plot(tiempo, campo/H0,'m-',label='Campo')

plt.plot(tiempo, magnet_eq,'k-',label='Equilibrio')
#plt.plot(tiempo, magnet_bc_ch/max(magnet_bc_ch),  'b-',label='Aproximación bajo campo')
plt.plot(tiempo, magnet_ec_dif,'r-',label='Ecuacion diferencial')
plt.plot(tiempo, magnet_eq_serie_retraso,'g-',label='Serie')
#plt.plot(tiempo, campo/H0,m-',label='Campo')

plt.legend(loc='best')
f2.set_ylabel('Magnetizacion')
f2.set_xlabel('Tiempo (s)')

f3 = plt.subplot(3,1,3)
plt.plot(campo, magnet_eq,'k-',label='Equilibrio')
plt.plot(campo, magnet_eq_serie,'g-',label='Serie equilibrio')
plt.legend(loc='best')
f2.set_ylabel('Magnetizacion')
f2.set_xlabel('Tiempo (s)')
plt.show()

#%%