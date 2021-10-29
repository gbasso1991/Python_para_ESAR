#%%Calibracion de bobinas en cofiguracion Helmholtz
# Giuliano Basso
import numpy as np
import matplotlib.pyplot as plt

# %% Datos Medidos 28-Oct
f_1 = 10 #Hz
Hall_1 = np.array([17.20,34.40,51.60,67.80,82.40,97.20]) # mV
Ref_1 = np.array([1.012,2.030,3.050,4.080,5.080,6.100])  # V

f_2 = 300 #Hz 
Hall_2 = np.array([16.80,33.90,51.20,66.80,82.80,97.60]) #mV
Ref_2 = np.array([1.000,2.030,3.040,4.060,5.080,6.100]) # V

f_3 = 600 #Hz 
Hall_3 = np.array([16.80,34.20,51.20,67.60,74.00]) #mV
Ref_3 = np.array([1.000,2.030,3.040,4.060,4.560])# V
Ref_3_NL = 5.060 #V

f_4 = 900 #Hz 
Hall_4 = np.array([16.80,25.20,33.40,42.00,50.20,])#mV
Ref_4 = np.array([1.000,1.500,2.000,2.500,3.000])# V
Ref_4_NL = 3.110 #V 

f_5 = 1200 #Hz 
Hall_5 = np.array([16.90,21.30,25.50,29.90,34.00])#mV
Ref_5 = np.array([1.000,1.250,1.500,1.760,2.000])# V
Ref_5_NL = 2.50 #V

f_6 = 1500 #Hz 
Hall_6 = np.array([17.20,21.50,25.90,30.20])#mV
Ref_6 = np.array([1.000,1.260,1.500,1.760])# V
Ref_6_NL = 2.00

f_7 = 1800 #Hz 
Hall_7 = np.array([17.40,21.80,26.30])#mV
Ref_7 = np.array([1.000,1.250,1.510])# V
Ref_7_NL = 1.610 #V

f_8 = 100 # Hz
Hall_8 = np.array([0.31,0.84,1.7,8.20,10.60,17.00,24.60,34.20,41.20,50.60,57.20,66.80,73.60,82.40,90.00,100.4,106.0,114.8,124.0])
Ref_8 =  np.array([0.010,0.051,0.101,0.504,0.650,1.010,1.520,2.030,2.520,3.060,3.560,4.060,4.580,5.080,5.600,6.100,6.600,7.100,7.600])


#%% Procesamiento 
frecuencias = [f_1,f_8,f_2,f_3,f_4,f_5,f_6,f_7]
Hall = np.array([Hall_1,Hall_2,Hall_3,Hall_4,Hall_5,Hall_6,Hall_7,Hall_8],dtype='object')
Ref = np.array([Ref_1,Ref_2,Ref_3,Ref_4,Ref_5,Ref_6,Ref_7,Ref_8],dtype='object')

#%% Fact de conversion sonda Hall: 1V = 0.1 T 
F_mV_to_T = 0.0001
campo_T = [d*F_mV_to_T for d in Hall] #T
# %%
fig, ax = plt.subplots(figsize=(8,5))
for i in range(len(frecuencias)):
    #print(len(campo_T[i]),'\n',len(Ref[i]),'\n',frecuencias[i],'\n')
    plt.plot(Ref[i],campo_T[i],'o-',label=f'$f$: {frecuencias[i]} Hz' )

plt.legend()
plt.xlabel('Amplitud de señal de referencia (V)')
plt.ylabel('Campo (T)')
plt.grid()    
plt.show()


# %% Ajusto recta
from scipy.optimize import curve_fit
def ajuste(x,m,n):
    return m*x+n
#%%
campo_T_all = np.concatenate([campo_T[i] for i  in range(len(frecuencias))])
Ref_all = np.concatenate([Ref[i] for i in range(len(frecuencias))])

opt_param, pcov = curve_fit(ajuste,Ref_all,campo_T_all)
x = np.linspace(0,7.6,1000)
y = ajuste(x,opt_param[0],opt_param[1])

fig, ax = plt.subplots(figsize=(8,5))

plt.plot(Ref_all, campo_T_all,'o')
plt.plot(x,y,label='Ajuste lineal')
plt.annotate(f'$y = m\\cdot x $\nm: {opt_param[0]:.3e} T/V\nn: {opt_param[1]:.3e} T',
xy=(0.7*max(x),0.1*max(y)),
bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7))

plt.legend()
plt.xlabel('Referencia (V)')
plt.ylabel('Campo (T)')
plt.grid()
plt.title('Bobinas de Helmholtz\nCampo vs. amplitud de señal')
#plt.savefig('calibracion_Helmholtz.png', dpi=300,facecolor='w')
# %% 29 - Oct
opt_param, pcov = curve_fit(ajuste,Ref_8,Hall_8*F_mV_to_T)
x = np.linspace(0,7.6,1000)
y = ajuste(x,opt_param[0],opt_param[1])

fig, ax = plt.subplots(figsize=(8,5))

plt.plot(Ref_8,Hall_8*F_mV_to_T,'o',label=f'$f: {frecuencias[1]}$ Hz')
plt.plot(x,y,label='Ajuste lineal')
plt.annotate(f'$y = m \\cdot x + n $\nm: {opt_param[0]:.3e} T/V\nn: {opt_param[1]:.3e} T',
xy=(0.7*max(x),0.1*max(y)),
bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7))

plt.legend()
plt.xlabel('Amplitud de señal (V)')
plt.ylabel('Campo (T)')
plt.grid()
plt.title('Bobinas de Helmholtz\nCampo vs. amplitud de señal')
plt.savefig('calibracion_Helmholtz.png', dpi=300,facecolor='w')

# %%
