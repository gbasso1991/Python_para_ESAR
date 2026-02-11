#%% devourer.py 
#Giuliano Basso 
'''Para levantar archivos .txt del osciloscopio OWON
registra si tiene 2 Canales y en tal caso duplica los graficos'''
import os
import numpy as np
import matplotlib.pyplot as plt
from cal_captora import F_Vs_to_T_captora
from sklearn.metrics import r2_score
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.signal import find_peaks 
from scipy.fft import fft , ifft 
def sinusoide(t,A,B,C,D):
    '''Crea sinusoide con params: 
        A=offset, B=amp, C=frec, D=fase
    '''
    return(A + B*np.sin(2*np.pi*C*t + D))
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
def ajusta_seno(t,v,escala='V'):
    '''
    Calcula seeds y ajusta sinusoide via curve_fit
    Para calacular la frecuencia mide tiempo entre picos
    si la medida es en mV, debe aclararse en escala
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
    if escala=='mV':
        offset=offset/1000
        amplitud = amplitud/1000
    else:
        pass
    return offset, amplitud , frecuencia, fase
def ajuste_lineal(x,m,n):
    return m*x+n
#%% Carga de los archivos
rutas_de_carga=[]
cwd = os.getcwd()
path = './data_220224/bobina_1_config53/'
for i in range(19): #numero de medidas en el directorio
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')

# Datos a ingresar por el usuario
delta_t = 1e-8 #s (El inverso de los 100MS/s, parametro del osciloscopio)
Idc_0 = 5 #A

frec = []
amp=[]
frec2 = []
amp2=[]
params=[]#offset,amplitud,frecuencia,fase 
params2=[]#offset,amplitud,frecuencia,fase
tiempos=[]
original1=[]
original2=[]
for j,e in enumerate(rutas_de_carga):
    medida = np.loadtxt(e,skiprows=3)
    t = (medida[:,0]-medida[0,0])*delta_t #1er columna, chequear base temporal 
    CH1 = np.array(medida[:,1],dtype='float') #(mV en gral)                       #V 
    v_1  = fft_smooth(CH1, np.around(int(len(CH1)*6/1000)))
    params.append(ajusta_seno(t,v_1,escala='mV'))
    original1.append(CH1)
    tiempos.append(t)
    if len(medida[0,:])>2:
        CH2 = np.array(medida[:,2],dtype='float') #CH2 (mV en gral) 
        v_2  = fft_smooth(CH2, np.around(int(len(CH2)*6/1000)))
        params2.append(ajusta_seno(t,v_2,escala='mV'))
        original2.append(CH2)


fig= plt.figure(figsize=(9,7))
ax0 = fig.add_subplot(211)
for i in range(len(tiempos)):
    ax0.plot(tiempos[i],original1[i]/1000,'-',lw=0.8,label=f'Original {i}')
#ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax0.grid()
plt.xlim(0,max(tiempos[i]))
plt.ylabel('Amplitud $(V)$')
plt.title('Señal en CH1')

ax1 = fig.add_subplot(212)
tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
for j in range(len(tiempos)):
    ax1.plot(tiempos[j],sinusoide(tiempos[j],params[j][0],params[j][1],params[j][2],params[j][3]),'-',lw=0.8,label=f'Simulada {j}')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax1.grid()
plt.xlim(0,max(tiempos_bis))
plt.xlabel('t $(s)$')
plt.ylabel('Amplitud $(V)$')
plt.show()
frec = np.array([elem[2] for elem in params])
amp = np.array([elem[1] for elem in params])
#Para señal en CH2 si hay
if original2:
    fig= plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(211)
    for i in range(len(tiempos)):
        ax0.plot(tiempos[i],original2[i]/1000,'-',lw=0.8,label=f'Original {i}')
    #ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
    ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax0.grid()
    plt.xlim(0,max(tiempos[i]))
    plt.ylabel('Amplitud $(V)$')
    plt.title('Señal en CH2')

    ax1 = fig.add_subplot(212)
    tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
    for j in range(len(tiempos)):
        ax1.plot(tiempos[j],sinusoide(tiempos[j],params2[j][0],params2[j][1],params2[j][2],params2[j][3]),'-',lw=0.8,label=f'Simulada {j}')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax1.grid()
    plt.xlim(0,max(tiempos_bis))
    plt.xlabel('t $(s)$')
    plt.ylabel('Amplitud $(V)$')
    plt.show()
    frec2 = np.array([elem[2] for elem in params2])
    amp2 = np.array([elem[1] for elem in params2])

# Ahora la coordenada z y el perfil
z= np.arange(0,25.4*len(frec)/40,25.4/40)
fig= plt.figure(figsize=(8,6))
plt.plot(z,amp,'-o',label=f'CH1 {np.mean(frec)/1000:.2f} +/- {np.std(frec)/1000:.2f} kHz')
#if original2:
    #plt.plot(z,amp2,'-o',label=f'CH2 {np.mean(frec2)/1000:.2f} +/- {np.std(frec2)/1000:.2f} kHz ')
plt.grid()
plt.xlabel('z ($mm$)')
plt.ylabel('Amplitud ($V$)')
plt.legend()
plt.title(f'Perfil de cancelacion dentro de la bobina generadora\nIdc: {Idc_0} A')
plt.savefig('perfil_cancelacion_0',dpi=300,facecolor='w')


##############################################################
# %% Again, con el otro set de medidas
rutas_de_carga=[]
cwd = os.getcwd()
path = './data_220224/bobina_1_config53_bis/'
for i in range(19): #numero de medidas en el directorio
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')

# Datos a ingresar por el usuario
delta_t = 1e-8 #s (El inverso de los 100MS/s, parametro del osciloscopio)
Idc_0 = 5 #A

frec = []
amp=[]
frec2 = []
amp2=[]
params=[]#offset,amplitud,frecuencia,fase 
params2=[]#offset,amplitud,frecuencia,fase
tiempos=[]
original1=[]
original2=[]
for j,e in enumerate(rutas_de_carga): #loop para los ajustes
    medida = np.loadtxt(e,skiprows=3)
    t = (medida[:,0]-medida[0,0])*delta_t #1er columna, chequear base temporal 
    CH1 = np.array(medida[:,1],dtype='float') #(mV en gral)                       #V 
    v_1  = fft_smooth(CH1, np.around(int(len(CH1)*6/1000)))
    params.append(ajusta_seno(t,v_1,escala='mV'))
    original1.append(CH1)
    tiempos.append(t)
    if len(medida[0,:])>2:
        CH2 = np.array(medida[:,2],dtype='float') #CH2 (mV en gral) 
        v_2  = fft_smooth(CH2, np.around(int(len(CH2)*6/1000)))
        params2.append(ajusta_seno(t,v_2,escala='mV'))
        original2.append(CH2)

fig= plt.figure(figsize=(9,7))
ax0 = fig.add_subplot(211)
for i in range(len(tiempos)):
    ax0.plot(tiempos[i],original1[i]/1000,'-',lw=0.8,label=f'Original {i}')
#ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax0.grid()
plt.xlim(0,max(tiempos[i]))
plt.ylabel('Amplitud $(V)$')

plt.title('Señal en CH1')
ax1 = fig.add_subplot(212)
tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
for j in range(len(tiempos)):
    ax1.plot(tiempos[j],sinusoide(tiempos[j],params[j][0],params[j][1],params[j][2],params[j][3]),'-',lw=0.8,label=f'Simulada {j}')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax1.grid()
plt.xlim(0,max(tiempos_bis))
plt.xlabel('t $(s)$')
plt.ylabel('Amplitud $(V)$')
plt.show()
frec = np.array([elem[2] for elem in params])
amp = np.array([elem[1] for elem in params])
fase = np.array([elem[3] for elem in params])
for i,elem in enumerate(fase):
    if elem>= np.pi:
        amp[i]=-amp[i]
#Para señal en CH2 si hay
if original2:
    fig= plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(211)
    for i in range(len(tiempos)):
        ax0.plot(tiempos[i],original2[i]/1000,'-',lw=0.8,label=f'Original {i}')
    #ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
    ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax0.grid()
    plt.xlim(0,max(tiempos[i]))
    plt.ylabel('Amplitud $(V)$')

    plt.title('Señal en CH2')
    ax1 = fig.add_subplot(212)
    tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
    for j in range(len(tiempos)):
        ax1.plot(tiempos[j],sinusoide(tiempos[j],params2[j][0],params2[j][1],params2[j][2],params2[j][3]),'-',lw=0.8,label=f'Simulada {j}')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax1.grid()
    plt.xlim(0,max(tiempos_bis))
    plt.xlabel('t $(s)$')
    plt.ylabel('Amplitud $(V)$')
    plt.show()
    frec2 = np.array([elem[2] for elem in params2])
    amp2 = np.array([elem[1] for elem in params2])

# Ahora la coordenada z y el perfil
z= np.arange(0,25.4*len(frec)/40,25.4/40)
fig= plt.figure(figsize=(8,6))
plt.plot(z,amp,'-o',label=f'CH1 {np.mean(frec)/1000:.2f} +/- {np.std(frec)/1000:.2f} kHz')
#if original2:
    #plt.plot(z,amp2,'-o',label=f'CH2 {np.mean(frec2)/1000:.2f} +/- {np.std(frec2)/1000:.2f} kHz ')
plt.grid()
plt.xlabel('z ($mm$)')
plt.ylabel('Amplitud ($V$)')
plt.legend()
plt.title(f'Perfil de cancelacion dentro de la bobina generadora\nIdc: {Idc_0} A')
plt.savefig('perfil_cancelacion_1',dpi=300,facecolor='w')

# %% Again,medidas 25 Feb 22
rutas_de_carga=[]
cwd = os.getcwd()
path = './data_220225/bobina_1_config53_5A/'
for i in range(19): #numero de medidas en el directorio
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')

# Datos a ingresar por el usuario
delta_t = 1e-8 #s (El inverso de los 100MS/s, parametro del osciloscopio)
Idc_0 = 5 #A

frec = []
amp=[]
frec2 = []
amp2=[]
params=[]#offset,amplitud,frecuencia,fase 
params2=[]#offset,amplitud,frecuencia,fase
tiempos=[]
original1=[]
original2=[]
for j,e in enumerate(rutas_de_carga): #loop para los ajustes
    medida = np.loadtxt(e,skiprows=3)
    t = (medida[:,0]-medida[0,0])*delta_t #1er columna, chequear base temporal 
    CH1 = np.array(medida[:,1],dtype='float') #(mV en gral)                       #V 
    v_1  = fft_smooth(CH1, np.around(int(len(CH1)*6/1000)))
    params.append(ajusta_seno(t,v_1,escala='mV'))
    original1.append(CH1)
    tiempos.append(t)
    if len(medida[0,:])>2:
        CH2 = np.array(medida[:,2],dtype='float') #CH2 (mV en gral) 
        v_2  = fft_smooth(CH2, np.around(int(len(CH2)*6/1000)))
        params2.append(ajusta_seno(t,v_2,escala='mV'))
        original2.append(CH2)

fig= plt.figure(figsize=(9,7))
ax0 = fig.add_subplot(211)
for i in range(len(tiempos)):
    ax0.plot(tiempos[i],original1[i]/1000,'-',lw=0.8,label=f'Original {i}')
#ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax0.grid()
plt.xlim(0,max(tiempos[i]))
plt.ylabel('Amplitud $(V)$')

plt.title('Señal en CH1')
ax1 = fig.add_subplot(212)
tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
for j in range(len(tiempos)):
    ax1.plot(tiempos[j],sinusoide(tiempos[j],params[j][0],params[j][1],params[j][2],params[j][3]),'-',lw=0.8,label=f'Simulada {j}')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax1.grid()
plt.xlim(0,max(tiempos_bis))
plt.xlabel('t $(s)$')
plt.ylabel('Amplitud $(V)$')

frec = np.array([elem[2] for elem in params])
amp = np.array([elem[1] for elem in params])
fase=np.array([elem[3] for elem in params])
plt.savefig('señal_CH1_orig_y_simulada_220225_5A',dpi=300,facecolor='w',bbox_inches='tight')

for idx,elem in enumerate(fase):
    if elem>= np.pi:
        break
amp[idx:]=-amp[idx:]
        
#Para señal en CH2 si hay
if original2:
    fig= plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(211)
    for i in range(len(tiempos)):
        ax0.plot(tiempos[i],original2[i]/1000,'-',lw=0.8,label=f'Original {i}')
    #ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
    ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax0.grid()
    plt.xlim(0,max(tiempos[i]))
    plt.ylabel('Amplitud $(V)$')

    plt.title('Señal en CH2')
    ax1 = fig.add_subplot(212)
    tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
    for j in range(len(tiempos)):
        ax1.plot(tiempos[j],sinusoide(tiempos[j],params2[j][0],params2[j][1],params2[j][2],params2[j][3]),'-',lw=0.8,label=f'Simulada {j}')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax1.grid()
    plt.xlim(0,max(tiempos_bis))
    plt.xlabel('t $(s)$')
    plt.ylabel('Amplitud $(V)$')

    frec2 = np.array([elem[2] for elem in params2])
    amp2 = np.array([elem[1] for elem in params2])
    fase2=np.array([elem[3] for elem in params2])
    plt.savefig('señal_CH2_orig_y_simulada_220225_5A',dpi=300,facecolor='w',bbox_inches='tight')

# Ahora la coordenada z y el perfil
z= np.arange(0,25.4*len(frec)/40,25.4/40)
fig= plt.figure(figsize=(8,6))
plt.plot(z,amp,'-o',label=f'CH1 {np.mean(frec)/1000:.2f} +/- {np.std(frec)/1000:.2f} kHz')
#if original2:
    #plt.plot(z,amp2,'-o',label=f'CH2 {np.mean(frec2)/1000:.2f} +/- {np.std(frec2)/1000:.2f} kHz ')
plt.axhline(0,0,1,c='k',lw=0.8)
plt.axvline(max(z)/2,0,1,c='k',lw=0.8)
plt.grid()
plt.xlabel('z ($mm$)')
plt.ylabel('Amplitud ($V$)')
plt.legend()
plt.title(f'Perfil de cancelacion dentro de la bobina generadora\nIdc: {Idc_0} A')
plt.savefig('perfil_cancelacion_220225_5A',dpi=300,facecolor='w',bbox_inches='tight')
#plt.show()
# %% Again,medidas 25 Feb 22
rutas_de_carga=[]
cwd = os.getcwd()
path = './data_220225/bobina_1_config53_10A/'
for i in range(19): #numero de medidas en el directorio
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')

# Datos a ingresar por el usuario
delta_t = 1e-8 #s (El inverso de los 100MS/s, parametro del osciloscopio)
Idc_0 = 10 #A

frec = []
amp=[]
frec2 = []
amp2=[]
params=[]#offset,amplitud,frecuencia,fase 
params2=[]#offset,amplitud,frecuencia,fase
tiempos=[]
original1=[]
original2=[]
for j,e in enumerate(rutas_de_carga): #loop para los ajustes
    medida = np.loadtxt(e,skiprows=3)
    t = (medida[:,0]-medida[0,0])*delta_t #1er columna, chequear base temporal 
    CH1 = np.array(medida[:,1],dtype='float') #(mV en gral)                       #V 
    v_1  = fft_smooth(CH1, np.around(int(len(CH1)*6/1000)))
    params.append(ajusta_seno(t,v_1,escala='mV'))
    original1.append(CH1)
    tiempos.append(t)
    if len(medida[0,:])>2:
        CH2 = np.array(medida[:,2],dtype='float') #CH2 (mV en gral) 
        v_2  = fft_smooth(CH2, np.around(int(len(CH2)*6/1000)))
        params2.append(ajusta_seno(t,v_2,escala='mV'))
        original2.append(CH2)

fig= plt.figure(figsize=(9,7))
ax0 = fig.add_subplot(211)
for i in range(len(tiempos)):
    ax0.plot(tiempos[i],original1[i]/1000,'-',lw=0.8,label=f'Original {i}')
#ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax0.grid()
plt.xlim(0,max(tiempos[i]))
plt.ylabel('Amplitud $(V)$')

plt.title('Señal en CH1')
ax1 = fig.add_subplot(212)
tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
for j in range(len(tiempos)):
    ax1.plot(tiempos[j],sinusoide(tiempos[j],params[j][0],params[j][1],params[j][2],params[j][3]),'-',lw=0.8,label=f'Simulada {j}')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax1.grid()
plt.xlim(0,max(tiempos_bis))
plt.xlabel('t $(s)$')
plt.ylabel('Amplitud $(V)$')
#plt.show()
plt.savefig('señal_CH1_orig_y_simulada_220225_10A',dpi=300,facecolor='w',bbox_inches='tight')

frec = np.array([elem[2] for elem in params])
amp = np.array([elem[1] for elem in params])
fase=np.array([elem[3] for elem in params])
for idx,elem in enumerate(fase):
    if elem>= np.pi:
        break
amp[idx:]=-amp[idx:]
        
#Para señal en CH2 si hay
if original2:
    fig= plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(211)
    for i in range(len(tiempos)):
        ax0.plot(tiempos[i],original2[i]/1000,'-',lw=0.8,label=f'Original {i}')
    #ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
    ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax0.grid()
    plt.xlim(0,max(tiempos[i]))
    plt.ylabel('Amplitud $(V)$')

    plt.title('Señal en CH2')
    ax1 = fig.add_subplot(212)
    tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
    for j in range(len(tiempos)):
        ax1.plot(tiempos[j],sinusoide(tiempos[j],params2[j][0],params2[j][1],params2[j][2],params2[j][3]),'-',lw=0.8,label=f'Simulada {j}')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax1.grid()
    plt.xlim(0,max(tiempos_bis))
    plt.xlabel('t $(s)$')
    plt.ylabel('Amplitud $(V)$')
    #plt.show()
    frec2 = np.array([elem[2] for elem in params2])
    amp2 = np.array([elem[1] for elem in params2])
    fase2=np.array([elem[3] for elem in params2])
    plt.savefig('señal_CH2_orig_y_simulada_220225_10A',dpi=300,facecolor='w',bbox_inches='tight')

# Ahora la coordenada z y el perfil
z= np.arange(0,25.4*len(frec)/40,25.4/40)
fig= plt.figure(figsize=(8,6))
plt.plot(z,amp,'-o',label=f'CH1 {np.mean(frec)/1000:.2f} +/- {np.std(frec)/1000:.2f} kHz')
#if original2:
    #plt.plot(z,amp2,'-o',label=f'CH2 {np.mean(frec2)/1000:.2f} +/- {np.std(frec2)/1000:.2f} kHz ')
plt.axhline(0,0,1,c='k',lw=0.8)
plt.axvline(max(z)/2,0,1,c='k',lw=0.8)
plt.grid()
plt.xlabel('z ($mm$)')
plt.ylabel('Amplitud ($V$)')
plt.legend()
plt.title(f'Perfil de cancelacion dentro de la bobina generadora\nIdc: {Idc_0} A')
plt.savefig('perfil_cancelacion_220225_10A',dpi=300,facecolor='w',bbox_inches='tight')
#plt.show()
# %% Again,medidas 25 Feb 22
rutas_de_carga=[]
cwd = os.getcwd()
path = './data_220225/bobina_1_config53_15A/'
for i in range(19): #numero de medidas en el directorio
    rutas_de_carga.append(path+'medida_'+str(i)+'.txt')

# Datos a ingresar por el usuario
delta_t = 1e-8 #s (El inverso de los 100MS/s, parametro del osciloscopio)
Idc_0 = 15 #A

frec = []
amp=[]
frec2 = []
amp2=[]
params=[]#offset,amplitud,frecuencia,fase 
params2=[]#offset,amplitud,frecuencia,fase
tiempos=[]
original1=[]
original2=[]
for j,e in enumerate(rutas_de_carga): #loop para los ajustes
    medida = np.loadtxt(e,skiprows=3)
    t = (medida[:,0]-medida[0,0])*delta_t #1er columna, chequear base temporal 
    CH1 = np.array(medida[:,1],dtype='float') #(mV en gral)                       #V 
    v_1  = fft_smooth(CH1, np.around(int(len(CH1)*6/1000)))
    params.append(ajusta_seno(t,v_1,escala='mV'))
    original1.append(CH1)
    tiempos.append(t)
    if len(medida[0,:])>2:
        CH2 = np.array(medida[:,2],dtype='float') #CH2 (mV en gral) 
        v_2  = fft_smooth(CH2, np.around(int(len(CH2)*6/1000)))
        params2.append(ajusta_seno(t,v_2,escala='mV'))
        original2.append(CH2)

fig= plt.figure(figsize=(9,7))
ax0 = fig.add_subplot(211)
for i in range(len(tiempos)):
    ax0.plot(tiempos[i],original1[i]/1000,'-',lw=0.8,label=f'Original {i}')
#ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax0.grid()
plt.xlim(0,max(tiempos[i]))
plt.ylabel('Amplitud $(V)$')

plt.title('Señal en CH1')
ax1 = fig.add_subplot(212)
tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
for j in range(len(tiempos)):
    ax1.plot(tiempos[j],sinusoide(tiempos[j],params[j][0],params[j][1],params[j][2],params[j][3]),'-',lw=0.8,label=f'Simulada {j}')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
ax1.grid()
plt.xlim(0,max(tiempos_bis))
plt.xlabel('t $(s)$')
plt.ylabel('Amplitud $(V)$')
plt.savefig('señal_CH1_orig_y_simulada_220225_15A',dpi=300,facecolor='w',bbox_inches='tight')

#plt.show()

frec = np.array([elem[2] for elem in params])
amp = np.array([elem[1] for elem in params])
fase=np.array([elem[3] for elem in params])
for idx,elem in enumerate(fase):
    if elem>= np.pi:
        break
amp[idx:]=-amp[idx:]
        
#Para señal en CH2 si hay
if original2:
    fig= plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(211)
    for i in range(len(tiempos)):
        ax0.plot(tiempos[i],original2[i]/1000,'-',lw=0.8,label=f'Original {i}')
    #ax0.legend(loc='upper left',bbox_to_anchor=(0.,-0.15),ncol=3)
    ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax0.grid()
    plt.xlim(0,max(tiempos[i]))
    plt.ylabel('Amplitud $(V)$')

    plt.title('Señal en CH2')
    ax1 = fig.add_subplot(212)
    tiempos_bis=np.linspace(0,max(tiempos[0]),2*len(tiempos[0]))
    for j in range(len(tiempos)):
        ax1.plot(tiempos[j],sinusoide(tiempos[j],params2[j][0],params2[j][1],params2[j][2],params2[j][3]),'-',lw=0.8,label=f'Simulada {j}')
    ax1.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=2)
    ax1.grid()
    plt.xlim(0,max(tiempos_bis))
    plt.xlabel('t $(s)$')
    plt.ylabel('Amplitud $(V)$')
    #plt.show()
    frec2 = np.array([elem[2] for elem in params2])
    amp2 = np.array([elem[1] for elem in params2])
    fase2=np.array([elem[3] for elem in params2])
    plt.savefig('señal_CH2_orig_y_simulada_220225_15A',dpi=300,facecolor='w',bbox_inches='tight')

# Ahora la coordenada z y el perfil
z= np.arange(0,25.4*len(frec)/40,25.4/40)
fig= plt.figure(figsize=(8,6))
plt.plot(z,amp,'-o',label=f'CH1 {np.mean(frec)/1000:.2f} +/- {np.std(frec)/1000:.2f} kHz')
#if original2:
    #plt.plot(z,amp2,'-o',label=f'CH2 {np.mean(frec2)/1000:.2f} +/- {np.std(frec2)/1000:.2f} kHz ')
plt.axhline(0,0,1,c='k',lw=0.8)
plt.axvline(max(z)/2,0,1,c='k',lw=0.8)
plt.grid()
plt.xlabel('z ($mm$)')
plt.ylabel('Amplitud ($V$)')
plt.legend()
plt.title(f'Perfil de cancelacion dentro de la bobina generadora\nIdc: {Idc_0} A')
plt.savefig('perfil_cancelacion_220225_15A',dpi=300,facecolor='w',bbox_inches='tight')
#plt.show()
# %%

# %%

# %%
