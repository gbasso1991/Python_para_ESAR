#%% cal_ESAR.py
# Giuliano Basso


import numpy as np
import matplotlib.pyplot as plt
from cal_captora import F_Vs_to_T_captora
from sklearn.metrics import r2_score
from uncertainties import ufloat
from scipy.optimize import curve_fit

#%%
print('''07 Marzo 2023 
Actualizo valores medidos en sonda hall 1V = 0.1 T luego de que calibracion en VSM arroje diferencia del %21
''')

#%% Datos medidos 11 Feb 22
# Bobina BI (la que esta sin el plastico y la bob de referencia)
# z ~ 90 mm de recorrido total 
#1º V from bottom to top - 28 samples
Idc_0 = 15 #A
frec_0 = 300e3 #Hz
fem_0 = np.array([11.4,14.34,18.5,22.5,29.1,35.4,42.8,48.8,53.0,55.0,57.60,58.6,58.2,57.20,54.4,49.20,42.8,33.6,24.4,16.8,11.2,7.7,5.3])#v
fem_0 = fem_0/2 #220222
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
fem_1 = fem_1/2 #220222
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
fem_2 = fem_2/2 #220222
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
fem_3 = fem_3/2 #220222
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
fem_4 = fem_4/2 #220222

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
z_new = np.linspace(0,39,5000)


campo_max = max(new(z_new))
z_campo_max = z_new[np.nonzero(new(z_new)==max(new(z_new)))]

campo_90 = 0.9*campo_max
z_min = z_new[np.nonzero(new(z_new).round(4)==campo_90.round(4))[0][0]]
z_max = z_new[np.nonzero(new(z_new).round(4)==campo_90.round(4))[0][-1]]
# z_edge = z_new[np.nonzero(new(z_new)==min(campo_bottom_to_top_4))][0]
campo_edge = 0.5*(new(0)+new(39))
#%%
#Paso de T a A/m 
campo_bottom_to_top_4_Am = campo_bottom_to_top_4/(4*np.pi*1e-7)
campo_max_Am = campo_max/(4*np.pi*1e-7)
campo_90_Am = campo_90/(4*np.pi*1e-7)
campo_edge_Am = campo_edge/(4*np.pi*1e-7)
new_Am = new(z_new)/(4*np.pi*1e-7)

fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
ax.plot(z_4,campo_bottom_to_top_4_Am/1e3,'o-',label=f'Idc: {Idc_4} $A$\n$f$: {frec_4/1e3} kHz')
ax.plot(z_new,new_Am/1e3,lw=0.9,label='Interpolacion\ncubica')

plt.axvline(recorrido_4/2,-1,1,linestyle='--',lw=0.8,c='k',label='Centro\ngeométrico')

plt.hlines (campo_max_Am/1e3,-1,z_campo_max,color='r',label=f'Campo max\n{campo_max_Am/1e3:.2f} $kA/m$')
plt.hlines(campo_edge_Am/1e3,-1,max(z_new),color='b',lw=.9,label=f'Campo en borde\n{campo_edge_Am/1e3:.2f} $kA/m$ ({100*campo_edge/campo_max:.0f} %)')

plt.hlines(campo_90_Am/1e3,-1,z_max,color='g',lw=.9,label=f'Campo 90%\n{campo_90_Am/1e3:.2f} $kA/m$')
plt.vlines(z_campo_max,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_max_Am/1e3,linestyle='-',color='r')
plt.vlines(z_min,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_90_Am/1e3,linestyle='-',color='g',lw=.9,label=f'z min\n{z_min:.1f} $mm$')
plt.vlines(z_max,ymin=min(campo_bottom_to_top_4_Am)/1e3,ymax=campo_90_Am/1e3,linestyle='-',color='g',lw=.9,label=f'z max\n{z_max:.1f} $mm$')
plt.axvspan(z_min,z_max,color='tab:green',alpha=0.3,zorder=-2,label=f'zona óptima\n{z_max-z_min:.0f} $mm$')


plt.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=1)
plt.grid()
plt.xlim(-1,40)
# plt.ylim(24.7,48)
plt.ylabel(' $H$ ($kA/m$)')
plt.xlabel('z ($mm$)')
# plt.title('H(z) - bobina captora descendente',fontsize=14,y=0.93)
#plt.savefig('Perfil_bobina_generadora_kAm.png',dpi=300,facecolor='w',bbox_inches='tight')

#%% Mismo grafico, en mT para comparar con calibracion de Nacho

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(z_4,campo_bottom_to_top_4*1e3,'o-',label=f'Idc: {Idc_4} $A$\n$f$: {frec_4/1e3} kHz')
ax.plot(z_new,new(z_new)*1e3,lw=0.9,label='Interpolacion\ncubica')
plt.axvspan(z_min,z_max,color='tab:green',alpha=0.3,zorder=-2,label=f'zona optima:\n{z_max-z_min:.0f} $mm$')
plt.axvline(recorrido_4/2,0,1,linestyle='--',lw=0.8,c='k',label='Centro\ngeometrico')
plt.hlines (campo_max*1e3,-1,z_campo_max,color='r',label=f'Campo max\n{campo_max*1e3:.2f} $mT$')
plt.hlines(campo_90*1e3,-1,z_max,color='g',lw=.9,label=f'Campo 90%\n{campo_90*1e3:.2f} $mT$')
plt.hlines(campo_edge*1e3,-1,max(z_new),color='b',lw=.9,label=f'Campo en borde\n{campo_edge*1e3:.2f} $mT$ ({100*campo_edge/campo_max:.2f} %)')
plt.vlines(z_campo_max,ymin=min(campo_bottom_to_top_4)*1e3,ymax=campo_max*1e3,linestyle='-',color='r')
plt.vlines(z_min,ymin=min(campo_bottom_to_top_4)*1e3,ymax=campo_90*1e3,linestyle='-',color='g',lw=.9,label=f'z min\n{z_min:.1f} $mm$')
plt.vlines(z_max,ymin=min(campo_bottom_to_top_4)*1e3,ymax=campo_90*1e3,linestyle='-',color='g',lw=.9,label=f'z max\n{z_max:.1f} $mm$')
plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=4)
plt.grid()
plt.xlim(-1,40)
# plt.ylim(30,60)
plt.ylabel('Campo ($mT$)')
plt.xlabel('z ($mm$)')
plt.title('Perfil de amplitud de campo en bobina de trabajo\nCaptora descendiente',fontsize=14)
#plt.savefig('Perfil_bobina_generadora_mT.png',dpi=300,facecolor='w',bbox_inches='tight')

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
fem_5 = fem_5/2 #220222
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

#%%
# Datos medidos el 14 Feb 22
# Bobina Captora ascendente desde el centro, luego descendente
# Avanzo de a 1/4 de escala del micrometro
# Captora comienza en borde inferior de la generadora 
# Captora termina en borde superior de la generadora
#7º V from center to top 
Idc_6 = 15 #A
frec_6 = 300e3 #Hz
fem_6 = np.array([54.0,55.2,55.2,55.6,56.0,56.0,55.6,55.2,55.89,54.4,54.8,54.8,54.8,54.0,54.0,53.6,53.2])#v
fem_6 = fem_6/2 #220222

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


# %% Datos medidos el 22 Feb 22
# Osciloscopio OWON 
Idc_7 = np.array([2,3,4,5,6,7,8,9,10,11,12]) #A
frec_7 = 300e3 #Hz
fem_7 = np.array([8.8,12.8,16.6,20.2,24.0,27.8,31.6,35.4,39.0,43.4,47.0])
campo_centro_T = fem_7*F_Vs_to_T_captora.nominal_value/frec_7
campo_centro_Am = campo_centro_T/(4*np.pi*1e-7) 

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(Idc_7,campo_centro_Am/1e3,'o-',label=f'$f$: {frec_7/1e3} kHz')
#plt.axvspan(10,47.87,color='#B87333',alpha=0.6)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('Idc ($A$)')
plt.title('Campo en bobina de trabajo vs. Idc',fontsize=14)
plt.show()
# %% forma sofisticada, saco las amplitudes y frecuencias de los archivos
from scipy.signal import find_peaks 
from scipy.optimize import curve_fit
from scipy.fft import fft , ifft 
def sinusoide(t,A,B,C,D):
    '''
    Crea sinusoide con params: 
        A=offset, B=amp, C=frec, D=fase
    '''
    return(A + B*np.sin(2*np.pi*C*t - D))
def fft_smooth(data_v,freq_n):
    """
    fft low pass filter para suavizar la señal. 
    data_v: datos a filtrar (array)
    frec_n: numero N de armonicos que conservo: primeros N 
    y ultimos N. 
    """
    fft_data_v = fft(np.array(data_v))
    s_fft_data_v = np.zeros(len(data_v),dtype=complex)
    s_fft_data_v[0:int(freq_n)] = fft_data_v[0:int(freq_n)]
    s_fft_data_v[-1-int(freq_n): ] = fft_data_v[-1-int(freq_n):] 
    s_data_v = np.real(ifft(s_fft_data_v))
    
    return s_data_v
def ajusta_seno(t,v):
    '''
    Calcula seeds y ajusta sinusoide via curve_fit
    Para calacular la frecuencia mide tiempo entre picos
    '''
    offset0 = v.mean() 
    amplitud0=(v.max()-v.min())/2

    indices, _ = find_peaks(v,height=0)
    t_entre_max = np.mean(np.diff(t[indices]))
    frecuencia0 = 1 /t_entre_max
    
    fase0 = 2*np.pi*frecuencia0*t[indices[0]] - np.pi/2

    p0 = [offset0,amplitud0,frecuencia0,fase0]
    
    coeficientes, _ = curve_fit(sinusoide,t,v,p0=p0)
    
    offset=coeficientes[0]
    amplitud = coeficientes[1]
    frecuencia = coeficientes[2]
    fase = coeficientes[3]

    return offset, amplitud , frecuencia, fase

def ajuste(x,m,n):
    return m*x+n
#%%
rutas_de_carga=[]
path = './data/bobina_0_bis/'
for i in range(15):
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')
delta_t = 1e-8
Idc = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #A
frec = []
amp=[]
for j,e in enumerate(rutas_de_carga):
    medida = np.loadtxt(e,skiprows=3)
    t_0 = (medida[:,0]-medida[0,0])*delta_t    #1er columna 
    v_0 = medida[:,1]                        #V 
    v_1  = fft_smooth(v_0, np.around(int(len(v_0)*6/1000)))
    offset,amplitud,frecuencia,fase = ajusta_seno(t_0,v_1)
    #recuerdo que esto estaba en mV
    v_0=v_0/1000
    v_1=v_1/1000
    amplitud= amplitud/1000
    offset=offset/1000
    frec.append(frecuencia)
    amp.append(amplitud)
    plt.plot(t_0,v_0,'-',label=f'Señal original - Idc: {Idc[j]} $A$')
    plt.plot(t_0,v_1,'-',label=f'Amp: {amplitud:.2f} $V$  f: {frecuencia/1000:.2f} $kHz$ fase: {fase:.2f} $1/s$  offset: {offset:.3f} $V$')
    plt.xlim(0,10/frecuencia)
    plt.xlabel('t ($s$)')
    plt.ylabel('Amplitud ($V$)')
    plt.grid()
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=1)
    plt.title(f'{e[16:]}')
    plt.show()
    print(f'''    offset: {offset:.3f} V
    amplitud: {amplitud:.3f} V
    frecuencia: {frecuencia:.3f} Hz
    fase: {fase:.3f} rad''')
#%% Ajuste Lineal
campo_Am=[]
fig, ax = plt.subplots(figsize=(8,5))
for k, elem in enumerate(amp):
    Campo_T = elem*F_Vs_to_T_captora.nominal_value/frec[k]
    Campo_Am = (elem*F_Vs_to_T_captora.nominal_value/(frec[k]*4*np.pi*1e-7)) 
    campo_Am.append(Campo_Am)
    ax.plot(Idc[k],Campo_Am/1e3,'o',label=f'$f$: {frec[k]/1e3:.2f} kHz')
plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=4)
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('Idc ($A$)')
plt.title('Campo en bobina de trabajo vs. Idc',fontsize=14)
##plt.savefig('bobina_BI_Idc_vs_campo.png',dpi=300,facecolor='w',bbox_inches='tight')
plt.show()

pendiente = []
ordenada=[]
opt_param, cov_param = curve_fit(ajuste,Idc,campo_Am)
pendiente.append(ufloat(opt_param[0],np.sqrt(np.diag(cov_param))[0]))
ordenada.append(ufloat(opt_param[1],np.sqrt(np.diag(cov_param))[1]))
x = np.linspace(0,15,1000)
y = ajuste(x,opt_param[0],opt_param[1])/1000 
f = ufloat(np.mean(frec),np.std(frec))
#grafico con el ajuste
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(Idc,np.array(campo_Am)/1000,'o',label = f'$f:$ {f/1000:.2f} $kHz$')
plt.plot(x,y, label= 'Ajuste lineal')
plt.text(0.55,0.1,f'$y=m\cdot x + n$\n$m$: {np.mean(pendiente):.2e} $1/m $\n$n$: {np.mean(ordenada):.2e} $A/m$',fontsize=14,bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('Idc ($A$)')
plt.title('Campo en bobina de trabajo vs. Idc',fontsize=14)
#plt.savefig('bobina_BI_Idc_vs_campo.png',dpi=300,facecolor='w',bbox_inches='tight')
plt.show()

print(f'''Campo a frecuencia {f/1000:.2f} kHz e Idc maxima (15 A): {max(campo_Am)/1000:.2f} kA/m''')
#%% Grafico rapido 
#import seaborn as sns
#sns.set_theme(color_codes=True)
#f, ax = plt.subplots(figsize=(5, 6))
#sns.regplot(x=Idc,y=np.array(campo_Am),data=(Idc,np.array(campo_Am)), ax=ax)
# %% Ahora en config 53, a 98 kHz

rutas_de_carga2=[]
path2 = './data/bobina_0_config53/'

for i in range(15):
    rutas_de_carga2.append(path2+'medida_'+str(i)+'.txt')

delta_t = 1e-8
Idc2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #A
frec2 = []
amp2=[]
for j,e in enumerate(rutas_de_carga2):
    medida2 = np.loadtxt(e,skiprows=3)
    t_2 = (medida2[:,0]-medida2[0,0])*delta_t #1er columna 
    v_2 = medida2[:,1] #V 
    #suavizo para ajustar
    v_3  = fft_smooth(v_2, np.around(int(len(v_2)*6/1000)))
    offset2,amplitud2,frecuencia2,fase2 = ajusta_seno(t_2,v_3)
    #recuerdo que esto estaba en mV y paso a V
    v_2=v_2/1000
    v_3=v_3/1000
    amplitud2= amplitud2/1000
    offset2=offset2/1000
    frec2.append(frecuencia2)
    amp2.append(amplitud2)
    plt.plot(t_2,v_2,'-',label=f'Señal original - Idc: {Idc2[j]} $A$')
    plt.plot(t_2,v_3,'-',label=f'Amp: {amplitud2:.2f} $V$  f: {frecuencia2/1000:.2f} $kHz$ fase: {fase2:.2f} $1/s$  offset: {offset2:.3f} $V$')
    plt.xlim(0,5/frecuencia2)
    plt.xlabel('t ($s$)')
    plt.ylabel('Amplitud ($V$)')
    plt.grid()
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=1)
    plt.title(f'{e[16:]}')
    plt.show()
    print(f'''    offset: {offset2:.3f} V
    amplitud: {amplitud2:.3f} V
    frecuencia: {frecuencia2:.3f} Hz
    fase: {fase2:.3f} rad''')
#%%
styles =plt.style.use('default')
campo_Am2=[]
fig, ax = plt.subplots(figsize=(8,5))
for k, elem in enumerate(amp2):
    Campo_T2 = elem*F_Vs_to_T_captora.nominal_value/frec2[k]
    Campo_Am2 = (elem*F_Vs_to_T_captora.nominal_value/(frec2[k]*4*np.pi*1e-7)) 
    campo_Am2.append(Campo_Am2)
    ax.plot(Idc[k],Campo_Am2/1e3,'o',label=f'$f$: {frec[k]/1e3:.2f} kHz')
plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=4)
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('Idc ($A$)')
plt.title('Campo en bobina de trabajo vs. Idc',fontsize=14)
##plt.savefig('bobina_BI_Idc_vs_campo.png',dpi=300,facecolor='w',bbox_inches='tight')
plt.show()

pendiente2 = []
ordenada2=[]
opt_param, cov_param = curve_fit(ajuste,Idc2,campo_Am2)
pendiente2.append(ufloat(opt_param[0],np.sqrt(np.diag(cov_param))[0]))
ordenada2.append(ufloat(opt_param[1],np.sqrt(np.diag(cov_param))[1]))
x2 = np.linspace(0,15,1000)
y2 = ajuste(x,opt_param[0],opt_param[1])/1000 
f2 = ufloat(np.mean(frec2),np.std(frec2))
#grafico con el ajuste
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(Idc2,np.array(campo_Am2)/1000,'o',label = f'$f:$ {f2/1000:.2f} $kHz$')
plt.plot(x2,y2, label= 'Ajuste lineal')
plt.text(0.55,0.1,f'$y=m\cdot x + n$\n$m$: {np.mean(pendiente2):.2e} $1/m $\n$n$: {np.mean(ordenada2):.2e} $A/m$',fontsize=14,bbox=dict(boxstyle='round',color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Campo ($kA/m$)')
plt.xlabel('Idc ($A$)')
plt.title('Campo en bobina de trabajo vs. Idc',fontsize=14)
#plt.savefig('bobina_BI_Idc_vs_campo.png',dpi=300,facecolor='w',bbox_inches='tight')
plt.show()

print(f'''Campo a frecuencia {f2/1000:.2f} kHz e Idc maxima (15 A): {max(campo_Am2)/1000:.2f} kA/m''')


# %% Los 2 ajustes juntos
fig, ax = plt.subplots(figsize=(7,4),constrained_layout=True)
pend=np.mean(pendiente)
pend2=np.mean(pendiente2)
ord=np.mean(ordenada)
ord2=np.mean(ordenada2)

#pendientes de viejas calibraciones
m0 = 57.2/15
g0= x*m0
m1= 51.8/15
g1=x*m1
m2= 49.0/15 
g2=x*m2

ax.plot(x,y, label= f'$f=$ {f/1000:.2f} $kHz$')#\nm: {pend:.2f} $1/m$\nn: {ord:.2f} $A/m$')
ax.plot(x2,y2, label= f'$f=$ {f2/1000:.2f} $kHz$')#\nm: {pend2:.2f} $1/m$\nn: {ord2:.2f} $A/m$')
# ax.plot(x,g0, label= 'Ignacio BIVA')
# ax.plot(x,g1, label= 'Diego BI')
# ax.plot(x,g2, label= 'Gustavo BI')
plt.legend(loc='best',fontsize=12)

plt.text(0.55,0.13,f'$H= {pend.nominal_value:.1f}\,1/m \cdot Idc + {ord.nominal_value:.1f}\,A/m$',fontsize=10,bbox=dict(boxstyle='round',color='tab:blue',alpha=0.8),transform=ax.transAxes)
plt.text(0.55,0.05,f'$H= {pend2.nominal_value:.1f}\,1/m \cdot Idc + {ord2.nominal_value:.1f}\,A/m$',fontsize=10,bbox=dict(boxstyle='round',color='tab:orange',alpha=0.8),transform=ax.transAxes)

plt.grid()
plt.xlim(0,15)
plt.ylim(0,)
plt.xlabel('$Idc$ ($A$)')
plt.ylabel('$H$ ($kA/m$)')
# plt.title('Campo en bobina de trabajo vs. Idc')
plt.savefig('H_vs_Idc.png', dpi=300)

# %%
pendiente_HvsI = np.mean([pendiente[0].nominal_value, pendiente2[0].nominal_value])


ordenada_HvsI = np.mean([ordenada[0].nominal_value, ordenada2[0].nominal_value])


print(f'Pendiente H vs Idc: {pendiente_HvsI:.1f}\nOrdendada H vs Idc: {ordenada_HvsI:.1f}')




print('\nfuncionando ok al 7 Mar 2023')
print('Rechequeado el 11 Feb 2026')













# %%
