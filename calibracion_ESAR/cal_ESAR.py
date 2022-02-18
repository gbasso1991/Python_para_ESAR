#%% cal_ESAR.py
# Giuliano Basso
import numpy as np
import matplotlib.pyplot as plt
from cal_captora import F_Vs_to_T_captora
from sklearn.metrics import r2_score
from uncertainties import ufloat

#%% Datos medidos 11 Feb 22
# Bobina BI (la que esta sin el plastico y la bob de referencia)
# z ~ 90 mm de recorrido total 
#1º V from bottom to top - 28 samples
Idc_0 = 15 #A
frec_0 = 300e3 #Hz
fem_0 = np.array([11.4,14.34,18.5,22.5,29.1,35.4,42.8,48.8,53.0,55.0,57.60,58.6,58.2,57.20,54.4,49.20,42.8,33.6,24.4,16.8,11.2,7.7,5.3])#v
z_0 = np.linspace(0,90,num=len(fem_0)) #mm
#%%
# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_bottom_to_top_0 = fem_0*F_Vs_to_T_captora.nominal_value/frec_0

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_0,campo_bottom_to_top_0*1e3,'o-',label=f'Idc: {Idc_0} $A$\n$f$: {frec_0/1e3} kHz')
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora ascendente',fontsize=14)
plt.show()

print('12 Feb 22: Se asumieron medidas equiespaciadas')
# %%#%% Datos medidos 11 Feb 22
# Bobina BI
# coordenada y comienza a ~16 mm de la parte superior de la bobina
#              termina a ~16 mm por debajo
#  
#2º V from top to bottom - 48 samples
Idc_1 = 15 #A
frec_1 = 300e3 #Hz
fem_1 = np.array([6.7,8.5,10.8,12.9,16.10,20.0,24.75,28.7,33.0,38.0,42.8,46.4,49.6,51.2,54.8,55.2,57.2,56.8,58.0,58.4,57.6,57.4,55.6,54.4,53.6,52.4,50.8,48.4,46.4,44.4,40.8,38.0,34.4,30.8,27.60,24.8,22.2,20.0,17.68,15.4,13.80,12.6,11.0,10.0,9.1,8.2,7.3,6.5])#v
recorrido_1 = 90 #mm
z_1 = np.linspace(recorrido_1,0,num=len(fem_1)) #mm

# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_top_to_bottom_1 = fem_1*F_Vs_to_T_captora.nominal_value/frec_1

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_1,campo_top_to_bottom_1*1e3,'o-',label=f'Idc: {Idc_1} $A$\n$f$: {frec_0/1e3} kHz')
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendente',fontsize=14)
plt.show()

print('')
# %%
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_0,campo_bottom_to_top_0*1e3,'o-',label='Ascendente')
ax.plot(z_1,campo_top_to_bottom_1*1e3,'o-',label='Descendiente')
plt.axvspan(16,74,color='#B87333',alpha=0.6)

plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nComparacion',fontsize=14)
plt.show()

#%% Estimaciones de avance por 1/2 vuelta descendente
avance= np.array([2.26,2.08,1.86,2.14,1.62,2.62])
print(f'Avance medio por 1/2 vuelta ascendente: {avance.mean():.1f}+/-{avance.std():.1f} mm')
# %%#%% Datos medidos 11 Feb 22
# Bobina BI
# coordenada y comienza debajo de la espira central 
#              termina por encima de la espira central
# Acomodada a ojimetro y desplazada  
#3º V from top to bottom 
Idc_2 = 15 #A
frec_2 = 300e3 #Hz
fem_2 = np.array([54.8,55.6,56.60,56.80,57.80,56.4,57.4,57.20,56.40,56.60,54.8,54.4,54.2,54.00,53.60])#v
recorrido_2 = (25.4)*len(fem_2)/40 #mm
z_2 = np.linspace(recorrido_2,0,num=len(fem_2)) #mm#np.linspace(16,num=len(fem_2)) #mm

# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_top_to_bottom_2 = fem_2*F_Vs_to_T_captora.nominal_value/frec_2

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_2,campo_top_to_bottom_2*1e3,'o-',label=f'Idc: {Idc_2} $A$\n$f$: {frec_0/1e3} kHz')
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora  - Paso fino')
plt.show()

# %%
# Datos medidos el 14 Feb 22
# Bobina Captora ascendente
# Avanzo de a 1/2 vuelta
# Captora a 10 mm por encima de la generadora 
# captora termina 10 mm por debajo de la generadora
#4º V from bottom to top - 40 samples
Idc_3 = 15 #A
frec_3 = 300e3 #Hz
fem_3 = np.array([12.7,15.8,19.3,22.9,28.8,
33.4,39.0,43.2,47.2,50.4,
54.2,55.8,57.6,59.0,61.8,
62.2,60.8,60.6,59.8,59.6,
58.6,57.4,56.2,54.8,52.8,
51.0,49.6,45.0,42.0,39.4,
34.4,31.6,28.4,24.6,23.0,
19.6,17.8,16.0,13.8,12.8])#v
recorrido_3 = 57.87 #mm
z_3 = np.linspace(recorrido_3,0,num=len(fem_3)) #mm

# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_top_to_bottom_3 = fem_3*F_Vs_to_T_captora.nominal_value/frec_3

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_3,campo_top_to_bottom_3*1e3,'o-',label=f'Idc: {Idc_3} $A$\n$f$: {frec_3/1e3} kHz')
plt.axvspan(10,47.87,color='#B87333',alpha=0.6)
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendente',fontsize=14)
plt.show()

# %%
# Datos medidos el 14 Feb 22
# Bobina Captora ascendente
# Avanzo de a 1/2 vuelta
# Captora comienza en borde inferior de la generadora 
# Captora termina en borde superior de la generadora
#5º V from bottom to top - 40 samples
Idc_4 = 15 #A
frec_4 = 300e3 #Hz
fem_4 = np.array([33.0,37.4,39.4,42.6,44.2,47.8,
50.6,53.0,54.6,56.2,57.0,58.6,
59.4,60.0,60.2,60.2,59.8,59.0,
58.8,57.4,55.8,53.8,51.2,48.2,
45.0,41.4,37.0,32.4])#v
recorrido_4 = 39.0 #mm
z_4 = np.linspace(0,recorrido_4,num=len(fem_4)) #mm


# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_bottom_to_top_4 = fem_4*F_Vs_to_T_captora.nominal_value/frec_4
# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_4,campo_bottom_to_top_4*1e3,'o-',label=f'Idc: {Idc_4} $A$\n$f$: {frec_4/1e3} kHz')
plt.axvspan(0,39,color='#B87333',alpha=0.5)
plt.axvline(recorrido_4/2,0,1,linestyle='--',c='k')
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendente',fontsize=14)
plt.show()

#%% Interpolo _4 para obtener 
from scipy.interpolate import CubicSpline

new = CubicSpline(z_4,campo_bottom_to_top_4)
z_new = np.linspace(0,39,1000)

campo_max = max(new(z_new))
z_campo_max = z_new[np.nonzero(new(z_new)==max(new(z_new)))]

campo_90 = 0.9*campo_max
z_min = z_new[np.nonzero(new(z_new).round(4)==campo_90.round(3))[0][0]]
z_max = z_new[np.nonzero(new(z_new).round(4)==campo_90.round(3))[0][1]]
z_edge = z_new[np.nonzero(new(z_new)==min(campo_bottom_to_top_4))][0]
campo_edge = new(z_edge) 
#%%
#Paso de T a A/m 
campo_bottom_to_top_4_Am = campo_bottom_to_top_4/(4*np.pi*1e-7)
campo_max_Am = campo_max/(4*np.pi*1e-7)
campo_90_Am = campo_90/(4*np.pi*1e-7)
campo_edge_Am = campo_edge/(4*np.pi*1e-7)
new_Am = new(z_new)/(4*np.pi*1e-7)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(z_4,campo_bottom_to_top_4_Am/1e3,'o-',label=f'Idc: {Idc_4} $A$\n$f$: {frec_4/1e3} kHz')
ax.plot(z_new,new_Am/1e3,lw=0.9,label='Interpolacion')
plt.axvspan(0,39,color='#B87333',alpha=0.4)

plt.axvline(recorrido_4/2,0,1,linestyle='--',lw=0.8,c='k',label='Centro\ngeometrico')
plt.hlines (campo_max_Am/1e3,0,z_campo_max,color='r',label=f'Campo max\n{campo_max_Am/1e3:.2f} $kA/m$')
plt.hlines(campo_90_Am/1e3,0,z_max,color='g',lw=.9,label=f'Campo 90%\n{campo_90_Am/1e3:.2f} $kA/m$')
plt.hlines(campo_edge_Am/1e3,0,max(z_new),color='b',lw=.9,label=f'Campo en borde\n{campo_edge_Am/1e3:.2f} $kA/m$ ({100*campo_edge/campo_max:.2f} %)')


plt.vlines(z_campo_max,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_max_Am/1e3,linestyle='-',color='r')
plt.vlines(z_min,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_90_Am/1e3,linestyle='-',color='g',lw=.9,label=f'z min\n{z_min:.1f} $mm$')
plt.vlines(z_max,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_90_Am/1e3,linestyle='-',color='g',lw=.9,label=f'z max\n{z_max:.1f} $mm$')

plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=4)
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendiente',fontsize=14)
plt.savefig('Perfil_bobina_generadora.png',dpi=300,facecolor='w',bbox_inches='tight')






# %%
# Datos medidos el 14 Feb 22
# Bobina Captora ascendente desde el centro, luego descendente
# Avanzo de a 1/4 de escala del micrometro
# Captora comienza en borde inferior de la generadora 
# Captora termina en borde superior de la generadora
#6º V from bottom to top - 40 samples
Idc_5 = 15 #A
frec_5 = 300e3 #Hz
fem_5 = np.array([57.4,57.4,57.0,57.0,57.0,56.8,56.8,56.4])#v
recorrido_5 = 2.54 #mm
z_5 = np.linspace(0,recorrido_5,num=len(fem_5)) #mm

# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_top_to_bottom_5 = fem_5*F_Vs_to_T_captora.nominal_value/frec_5

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_5,campo_top_to_bottom_5*1e3,'o-',label=f'Idc: {Idc_4} $A$\n$f$: {frec_4/1e3} kHz')

plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendente - Paso fino')
plt.show()

# %%
# Datos medidos el 14 Feb 22
# Bobina Captora ascendente desde el centro, luego descendente
# Avanzo de a 1/4 de escala del micrometro
# Captora comienza en borde inferior de la generadora 
# Captora termina en borde superior de la generadora
#7º V from center to top 
Idc_6 = 15 #A
frec_6 = 300e3 #Hz
fem_6 = np.array([54.0,55.2,55.2,55.6,56.0,56.0,55.6,55.2,55.89,54.4,54.8,54.8,54.8,54.0,54.0,53.6,53.2])#v
recorrido_6 = 5.08 #mm
z_6 = np.linspace(recorrido_6,0,num=len(fem_6)) #mm

# Paso a campo fem (V) a campo magnetico (T=V*s/m^2) usando la frecuencia y la cte de la captora
campo_top_to_bottom_6 = fem_6*F_Vs_to_T_captora.nominal_value/frec_6

# Perfil de amplitud de campo B vs Z
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(z_6,campo_top_to_bottom_6*1e3,'o-',label=f'Idc: {Idc_1} $A$\n$f$: {frec_0/1e3} kHz')
#plt.axvspan(10,47.87,color='#B87333',alpha=0.6)
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendente',fontsize=14)
plt.show()