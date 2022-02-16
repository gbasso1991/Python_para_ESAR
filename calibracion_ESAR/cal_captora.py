#%% cal_captora.py
# Giuliano Basso
import numpy as np
import matplotlib.pyplot as plt
from cal_helmholtz import F_V_to_T_helmholtz
from sklearn.metrics import r2_score
import time
#%% Datos medidos el 10 Feb 2022   
f0=300 #Hz
ref0= np.array([1.01,2.033,3.04,4.02,5.02,6.02,7.028]) #V
fem0= np.array([1.524,2.313,2.36,3.14,3.88,4.70,5.38])*1e-3#V

f1=600#Hz 
ref1= np.array([1.014,2.02,3.04,4.028,4.33])# V
fem1= np.array([1.42,3.24,4.72,6.56,7.08])*1e-3#V

f2=900#Hz
ref2= np.array([1.023,1.505,2.02,2.502,3.007]) #V
fem2= np.array([2.34,3.72,4.76,6.28,7.36])*1e-3 #V

f3=1200#Hz
ref3= np.array([1.022,1.24,1.525,1.765,2.027]) #V
fem3= np.array([3.28,3.94,4.96,5.54,6.72])*1e-3 #V

f4=1500#Hz
ref4= np.array([0.864,1.0,1.24,1.509,1.77]) #V
fem4= np.array([3.48,4.08,5.00,6.24,7.32])*1e-3 #V

f5=1800#Hz
ref5= np.array([0.86,1.024,1.26,1.521]) #V
fem5= np.array([4.24,5.00,6.28,7.52])*1e-3 #V

f6=2100#Hz
ref6= np.array([0.763,0.86,1.025,1.274]) #V
fem6= np.array([4.44,5.00,5.88,7.36])*1e-3 #V


#%% Conversion de unidades - Utilizo factor obtenido en cal_helmholtz.py
F_T_to_Am = np.pi*4e-7# (A/m)/T

#%%
frec = np.array([f1,f2,f3,f4,f5,f6]) #Hz
campo_T = np.array([ref1,ref2,ref3,ref4,ref5,ref6],dtype=object)*F_V_to_T_helmholtz #T
fem = np.array([fem1,fem2,fem3,fem4,fem5,fem6],dtype=object) #V

#fem reducida == fem/frec 
fem_red = np.array([fem[i]/frec[i] for i in range(len(frec))] , dtype='object') #divido por la frecuencia

fig, ax = plt.subplots(figsize=(8,5))
for i,e in enumerate(campo_T):
    plt.plot(fem_red[i],e*1e3, 'o-',label=f'{frec[i]} Hz')

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('fem/f $(V\cdot s)$')
plt.ylabel('Campo $(mT)$')
plt.title('fem reducida en bobina captora vs. campo')
plt.grid()
plt.show()    

# %% Ajusto recta a cada serie de medidas
from scipy.optimize import curve_fit
def ajuste(x,m,n):
    return m*x+n
from uncertainties import ufloat
#%%

fig, ax = plt.subplots(figsize=(10,7))
pendiente = []
ordenada=[]

for i in range(0,len(frec)-1):
    opt_param, cov_param = curve_fit(ajuste,fem_red[i],campo_T[i])
    x = np.linspace(min(fem_red[i]),max(fem_red[i]),1000)
    y = ajuste(x,opt_param[0],opt_param[1]) 
    R = r2_score(campo_T[i],ajuste(fem_red[i],opt_param[0],opt_param[1]))
    pendiente.append(ufloat(opt_param[0],np.sqrt(np.diag(cov_param))[0]))
    ordenada.append(ufloat(opt_param[1],np.sqrt(np.diag(cov_param))[1]))

    #print(f'''$f: {frec[i]} Hz
#m: {opt_param[0]:.2f} +/- {np.sqrt(np.diag(cov_param))[0]:.2f} 1/m^2
#n: {opt_param[1]:.2e} +/- {np.sqrt(np.diag(cov_param))[1]:.2e} T
#----------------------------------------''')
    label = f'''$f:$ {frec[i]} Hz\nm: {opt_param[0]:.2e} $1/m^2$ \nn: {opt_param[1]:.2e} \n$R^2$: {R:.4f}\n'''
    plt.scatter(fem_red[i],campo_T[i])
    plt.plot(x,y,label=label)

plt.text(0.6,0.2,f'$y=m\cdot x + n$\n$m$: {np.mean(pendiente):.2e} $1/m^2$\n$n$: {np.mean(ordenada):.2e} $T$',fontsize=14,bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.xlim(0,)
plt.legend(loc='upper left', bbox_to_anchor=(0,-0.1),ncol=4)
plt.xlabel('fem reducida ($V \cdot s$)')
plt.ylabel('Campo ($T$)')
plt.grid()
plt.title('$fem/f$ vs. campo')
plt.suptitle('Calibración de bobina captora',fontsize=15)
plt.savefig('calibracion_Helmholtz.png',dpi=300,facecolor='w',bbox_inches='tight')
#plt.show()
#%%
pendiente = np.asarray(pendiente)
ordenada = np.asarray(ordenada)
F_Vs_to_T_captora = np.mean(pendiente) #1/m^2 
print(time.strftime("%d %b %Y - %H:%M:%S", time.localtime()))
print('-'*40)
print(f'''El factor de conversión de la fem reducida (V*s) al campo (T=V*s/m^2) para la bobina captora es:
          
          F_Vs_to_T_captora = {F_Vs_to_T_captora:.2e} 1/m^2.

Recordar que esto se realizó con un amplificador de señal tal que 1 V == 1 A en su rango de linealidad (o en eso confiamos).''')

#%%
