#%%Calibracion de bobinas en cofiguracion Helmholtz
# Giuliano Basso
# Nueva calibracion 10 Feb 2022
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
# %% Datos Medidos 10 Feb - Todos son valores amp
f_1 = 100 #Hz
Ref_1 = np.array([1.01,2.02,3.04,4.02,5.02,6.02])  # V
Hall_1 = np.array([16.4,33.4,50.0,66.4,82.4,98.59])*1e-3 # V

f_2 = 300 #Hz 
Ref_2 = np.array([1.01,2.02,3.04,4.0,5.02,6.02]) # V
Hall_2 = np.array([16.8,33.8,50.4,67.37,83.2,99.2])*1e-3 #V

f_3 = 600 #Hz 
Ref_3 = np.array([1.01,2.03,3.04,4.027,4.52])# V
Hall_3 = np.array([16.8,33.6,50.95,67.20,75.2])*1e-3 #V
 
f_4 = 900 #Hz 
Ref_4 = np.array([1.01,2.03,3.03])# V
Hall_4 = np.array([16.8,33.8,50.8])*1e-3#V

f_5 = 1200 #Hz 
Ref_5 = np.array([0.8,1.0,1.52,2.02])# V
Hall_5 = np.array([13.8,17.4,26.0,34.8])*1e-3#V

f_6 = 1500 #Hz 
Ref_6 = np.array([0.8,1.01,1.52])# V
Hall_6 = np.array([14.0,17.5,26.4])*1e-3#V

f_7 = 1800 #Hz 
Ref_7 = np.array([0.8,1.01,1.52])# V
Hall_7 = np.array([14.2,17.8,26.9])*1e-3#V

f_8 = 2100 # Hz
Ref_8 =  np.array([0.8,1.02,1.22])#V
Hall_8 = np.array([14.5,18.1,21.8])*1e-3 #V


#%% Procesamiento 
frecuencias = np.array([f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8])
Hall = np.array([Hall_1,Hall_2,Hall_3,Hall_4,Hall_5,Hall_6,Hall_7,Hall_8],dtype='object')
Ref = np.array([Ref_1,Ref_2,Ref_3,Ref_4,Ref_5,Ref_6,Ref_7,Ref_8],dtype='object')

#%% Fact de conversion sonda Hall: 1V = 0.1 T 
F_V_to_T = 0.1 
campo_T = Hall*F_V_to_T #paso mV a T
# %%
fig, ax = plt.subplots(figsize=(8,5))
for i in range(len(frecuencias)):
    #print(len(campo_T[i]),'\n',len(Ref[i]),'\n',frecuencias[i],'\n')
    plt.plot(Ref[i],campo_T[i]*1e3,'o-',label=f'$f$: {frecuencias[i]} Hz' )

plt.legend()
plt.xlabel('Amplitud de señal de referencia ($V$)\nAmplificador')
plt.ylabel('Campo ($mT$)\nSonda Hall ($1V = 0.1 T$)')
plt.title('Respuesta del par Helmholtz')
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
err_pend = np.sqrt(np.diag(pcov))[0]
err_ord = np.sqrt(np.diag(pcov))[1]

x = np.linspace(0,max(Ref_all),1000)
y = ajuste(x,opt_param[0],opt_param[1]) 

R = r2_score(campo_T_all,ajuste(Ref_all,opt_param[0],opt_param[1]))

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(Ref_all,campo_T_all,'.')
plt.plot(x,y,label='Ajuste lineal')
plt.annotate(f'$y = m \\cdot x + n $\nm: {opt_param[0]:.2e} +/- {err_pend:.2e} T/V\nn: {opt_param[1]:.2e} +/- {err_ord:.2e} T\n$R^2=$ {R:.4f}',
xy=(0.6*max(x),0.1*max(y)),fontsize=12,
bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7))
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Amplitud de señal ($V$)')
plt.ylabel('Campo ($T$)')
plt.grid()
plt.title('Calibracion del par Helmholtz')
#plt.savefig('calibracion_Helmholtz.png', dpi=300,facecolor='w')
plt.show()
#%%
F_V_to_T_helmholtz = opt_param[0] #Tesla/V .

print(time.strftime("%d %b %Y - %H:%M:%S", time.localtime()))
print('-'*40)
print(f'''El factor de conversión Volts a Tesla para bobinas Helmholzt es: 

       F_V_to_T_helmholtz: {F_V_to_T_helmholtz:.3e} T/V

Recordar que esto se realizó con un amplificador de señal tal que 1 V == 1 A en su rango de linealidad''')


# %%

# %%
