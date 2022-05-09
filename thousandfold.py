#%% 
# -*- coding: utf-8 -*-
'''
thousandfold.py 
https://www.youtube.com/watch?v=kb8WGig0MLU

@author: Giuliano Andrés Basso

Script para levantar señales de fondo y de calibracion (Gd2O3) registradas con el osciloscopio OWON.
Levanto archivos .txt, los filtro por Fourier y reconstruyo con frecuencia fundamental.
Grafico todos los ciclos, todas las rectas de ajuste y reporto pendiente/ordenada del conjunto de medidas. 
'''
print(__doc__)
#%%
from funciones_de_possessor import * 

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

def ajusta_seno_2(t,v,escala='V'):
    '''
    Calcula params de inicializacion y ajusta sinusoide via curve_fit
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
    return offset, amplitud , frecuencia, fase

def ajuste_lineal(x,m,n):
    return m*x+n

# Fourier para romanticide (calibracion vs cal-fondo)
def fourier_señales_2(t_c,v_c,v_r_c,delta_t,filtro,frec_limite,name,polaridad=1):
    '''
    Toma señales de calibracion junto a referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos (usar funcion 'recorte()' previamente).
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se asume polaridad 1. 
    '''
    t_c = t_c - t_c[0] #Tiempo de la calibracion 
    t_r_c = t_c.copy() #Tiempo de la referencia
    
    y_c = polaridad*v_c #calibracion (magnetizacion del paramagneto)
    y_r_c = v_r_c      #referencia calibracion (campo)
    
    N_c = len(v_c)
    N_r_c = len(v_r_c)

    if len(t_c)<len(y_c): #alargo t
        t_c = np.pad(t_c,(0,delta_t*(len(y_c)-len(t_c))),mode='linear_ramp',end_values=(0,max(t_c)+delta_t*(len(y_c)-len(t_c))))
    elif len(t_c)>len(y_c):#recorto t    
        t_c=np.resize(t_c,len(y_c))
    
    #Idem referencia
    if len(t_r_c)<len(y_r_c): #alargo t
        t_r_c = np.pad(t_r_c,(0,delta_t*(len(y_r_c)-len(t_r_c))),mode='linear_ramp',end_values=(0,max(t_r_c)+delta_t*(len(y_r_c)-len(t_r_c))))
    elif len(t_r_c)>len(y_r_c):#recorto t    
        t_r_c=np.resize(t_r_c,len(y_r_c))

#Aplico transformada de Fourier
    #Calibracion
    f_c = rfftfreq(N_c, d=delta_t)
    f_c_HF = f_c.copy()
    f_c = f_c[np.nonzero(f_c<=frec_limite)]
    g_c_aux = fft(y_c,norm='forward') 
    g_c = abs(g_c_aux)
    fase_c= np.angle(g_c_aux)
    
    #Referencia de calibracion
    f_r_c = rfftfreq(N_r_c, d=delta_t)
    f_r_c = f_r_c[np.nonzero(f_r_c<=frec_limite)]
    g_r_c_aux = fft(y_r_c,norm='forward')
    g_r_c = abs(g_r_c_aux)
    fase_r_c = np.angle(g_r_c_aux)
     
    #Recorto vectores hasta frec_limite
    g_c_HF = g_c.copy()
    g_c = np.resize(g_c,len(f_c))
    g_r_c = np.resize(g_r_c,len(f_r_c))

    g_c_HF = np.resize(g_c_HF,len(f_c_HF))
    g_c_HF_aux=g_c_HF.copy()
    g_c_HF[np.argwhere(f_c_HF<=frec_limite)]=0 #Anulo LF

#Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)
    indices_c,_=find_peaks(abs(g_c),height=max(g_c)*filtro)#borre threshold 
    indices_r_c,_=find_peaks(abs(g_r_c),height=max(g_r_c)*filtro)

    print(name)
    print('-'*40)
#En caso de frecuencia anomala menor que la fundamental en Muestra
    for elem in indices_c:
        if f_c[elem]<0.95*f_r_c[indices_r_c[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de calibracion {:.2f} Hz\n'.format(f_c[elem]))
            indices_c = np.delete(indices_c,0)
            
    armonicos_c = f_c[indices_c]
    amplitudes_c = g_c[indices_c]
    fases_c = fase_c[indices_c]

    armonicos_r_c = f_r_c[indices_r_c]
    amplitudes_r_c = g_r_c[indices_r_c]
    fases_r_c = fase_r_c[indices_r_c]
#Imprimo tabla 

    print('''\nEspectro de la señal de calibracion:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_c)):
        print(f'{armonicos_c[i]:<10.2f}    {amplitudes_c[i]/max(amplitudes_c):>12.2f}    {fases_c[i]:>12.4f}')
    
    print('''Espectro de la señal de referencia:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_r_c)):
        print(f'{armonicos_r_c[i]:<10.2f}    {amplitudes_r_c[i]/max(amplitudes_r_c):>12.2f}    {fases_r_c[i]:>12.4f}')
    
    print('-'*40)


#Idem Calibracion
    h_c_aux_impar = np.zeros(len(f_c),dtype=np.cdouble)
    for W in indices_c:
        h_c_aux_impar[W]=g_c_aux[W]
    reconstruida_c = irfft(h_c_aux_impar,n=len(t_c),norm='forward')    

#Reconstruyo señal limitada con ifft
    g_c_aux = np.resize(g_c_aux,len(f_c))
    rec_limitada = irfft(g_c_aux,n=len(t_c),norm='forward')

#Reconstruyo señal de alta frecuencia
    rec_c_HF = irfft(g_c_HF,n=len(t_c),norm='forward')
#Resto HF a la señal original y comparo con reconstruida impar

#Veo que tanto se parecen
    #r_2 = r2_score(rec_impares,rec_limitada)
    #r_2_resta  = r2_score(rec_impares,resta)
    

#Grafico 1.1 (Calibracion): 
    fig4 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral',fontsize=20)
#Señal Orig + Ref
    ax1 = fig4.add_subplot(3,1,1)
    ax1.plot(t_c,y_c/max(y_c),'.-',lw=0.9,label='Calibracion')
    ax1.plot(t_r_c,y_r_c/max(y_r_c),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title(str(name)+'\nReferencia y calibracion', loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  
#Espectro de Frecuencias 
    ax2 = fig4.add_subplot(3,1,2)
    ax2.plot(f_c/1000,g_c,'.-',lw=0.9)
    ax2.scatter(armonicos_c/1000, amplitudes_c,c='r',label='Componentes principales')
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.axvline(armonicos_r_c/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax2.axhline(y=max(g_c)*filtro,xmin=0,xmax=1,c='tab:orange',label=f'Filtro ({filtro*100}%)')
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f_c)/1000)
    ax2.legend(loc='best')
#  Espectro de Fases 
    ax3 = fig4.add_subplot(3,1,3)
    ax3.vlines(armonicos_c/1000,ymin=0,ymax=fases_c)
    ax3.stem(armonicos_c/1000,fases_c,basefmt=' ')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axvline(armonicos_r_c/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f_c)/1000)
    ax3.grid(axis='y')
    ax3.legend()

#Grafico 2.1: Espectro Impar, Fasorial, Original+Rec_impar (Calibracion)
    fig5 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion limitando frecuencias (calibracion)',fontsize=20)
# Señal Original + Reconstruida 
    ax1=fig5.add_subplot(2,1,1)
    ax1.plot(t_c,y_c,'.-',lw=0.9,label='Señal original')
    ax1.plot(t_c,rec_limitada,'-',label='Señal filtrada')
    ax1.plot(t_c,reconstruida_c,'-',label='Señal reconstruida')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.set_ylim(-4*max(reconstruida_c),4*max(reconstruida_c))
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax1.grid() 
    ax1.legend(loc='best')

#Grafico 3.1: Altas frecuencias (calibracion)
    fig6 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Espectro y señal',fontsize=20)
# Espectro en fases impares
    ax1=fig6.add_subplot(2,1,1)
    ax1.plot(f_c_HF[f_c_HF<frec_limite]/1000,g_c_HF_aux[f_c_HF<frec_limite],'.-',lw=0.8,c='tab:blue',label='Espectro limitado')
    #ax1.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax1.stem(armonicos_c/1000,(amplitudes_c/max(amplitudes_c))*max(amplitudes_c),basefmt=' ',markerfmt='or',linefmt='--r',bottom=0.0,label='Frecuencias principales')
    ax1.legend(loc='best')
    ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    ax1.set_xlabel('Frecuencia (kHz)')
    ax1.set_ylabel('|F{$\epsilon$}|')   
    #ax1.set_xlim(0,max(f)/1000)
    #ax1.set_ylim(0,max(amp_impar)*1.1)
    ax1.grid()

# Señal HF + Reconstruida LF
    ax2=fig6.add_subplot(2,1,2)
    #ax2.plot(t_c,rec_c_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    ax2.plot(t_c,y_c,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1)    
    ax2.plot(t_c,rec_limitada,'-',lw=1,label='Frecuencia limitada',zorder=3)
    ax2.plot(t_c,reconstruida_c,'-',lw=1,c='tab:green',label='Reconstruccion',zorder=5)

    ax2.set_xlabel('t (s)')
    ax2.set_xlim(0,5/armonicos_c[0])
    ax2.set_ylim(1.1*min(y_c),1.1*max(y_c))
    ax2.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax2.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax2.grid() 
    ax2.legend(loc='best')

    return reconstruida_c,armonicos_c, armonicos_r_c, amplitudes_c, amplitudes_r_c, fases_c , fases_r_c , fig, indices,fig4,fig5,fig6


############################################################

###########################################################

############################################################

###########################################################

#%%
'''
Seleccion de carpeta con archivos via interfaz de usuario
'''
import tkinter as tk
from tkinter import filedialog
import os
import fnmatch
root = tk.Tk()
root.withdraw()

texto_encabezado = "Seleccionar la carpeta con las medidas a analizar:"
directorio = filedialog.askdirectory(title=texto_encabezado)
filenames = os.listdir(directorio) #todos

fnames_c = []
path_c = []
fnames_f = []
path_f = []

for cal in fnmatch.filter(filenames,'*_cal*'):
    fnames_c.append(cal)
    path_c.append(directorio + '/'+ cal)

for fondo in fnmatch.filter(filenames,'*_fondo*'):
    fnames_f.append(fondo)
    path_f.append(directorio + '/' + fondo)

print('Directorio de trabajo: '+ directorio +'\n')
print('Archivos de fondo: ')
for item in fnames_f:
    print(item)
print('\nArchivos de calibracion: ')
for item in fnames_c:
    print(item)


'''
Parámetros de la medida a partir de nombre del archivo 
de muestra: 'xxxkHz_yyydA_zzzMss_label.txt
'''
frec_nombre=[] #Frec del nombre del archivo. Luego comparo con frec ajustada
Idc = []       #Internal direct current en el generador de RF
delta_t_c = []   #Base temporal 
delta_t_f = [] 
labels_c = []

labels_f = []
for i in range(len(fnames_c)):
    frec_nombre.append(float(fnames_c[i].split('_')[0][:-3])*1000)
    Idc.append(float(fnames_c[i].split('_')[1][:-2])/10)
    delta_t_c.append(1e-6/float(fnames_c[i].split('_')[2][:-3]))
    delta_t_f.append(1e-6/float(fnames_f[i].split('_')[2][:-3]))
    labels_c.append(str(fnames_c[i].split('_')[3][:-4]))    
    labels_f.append(str(fnames_f[i].split('_')[3][:-4]))    

ciclos_H = []
ciclos_M = []
ciclos_M_ua = []
pendientes=[]
ordenadas=[]
for k in range(len(fnames_c)):
    '''defino DataFrames con los datos de muestra, calibracion y fondo'''
    df_c = medida_cruda(path_c[k],delta_t_c[k])
    df_f = medida_cruda(path_f[k],delta_t_f[k])
    '''
    Realizo el ajuste sobre la referencia y obtengo params
    '''
    offset_c , amp_c, frec_c , fase_c = ajusta_seno(df_c['t'],df_c['v_r'])
    offset_f , amp_f, frec_f , fase_f = ajusta_seno(df_f['t'],df_f['v_r'])

    #Genero señal simulada usando params y guardo en respectivos df
    df_c['v_r_ajustada'] = sinusoide(df_c['t'],offset_c , amp_c, frec_c , fase_c)
    df_f['v_r_ajustada'] = sinusoide(df_f['t'],offset_f , amp_f, frec_f , fase_f)

    '''Ejecuto la funcion resta_inter() '''

    #Arrays sobre los que laburo: Tiempo, Calibracion, Referencia de Calibracion y Fondo
    t_c = df_c['t'].to_numpy()
    v_c = df_c['v'].to_numpy()
    v_r_c = df_c['v_r'].to_numpy()
    v_f = df_f['v'].to_numpy()

    Resta_c , t_c_1 , v_r_c_1 , _ = resta_inter(t_c,v_c,v_r_c,fase_c,frec_c,offset_c,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)
 
    #Grafico las restas
    print('Grafico señal de calibracion, fondo y la resta entre ellas')
    fig, (ax,ax1) = plt.subplots(2,sharex=True,sharey=False,figsize=(8,6))
    fig.suptitle(fnames_c[k])
    ax.plot(t_c_1,v_c,'-',lw=0.9,label='Calibracion')
    ax.plot(t_c_1,v_f,'-',lw=0.5,label='Fondo',alpha=0.6)
    ax.grid()
    ax.legend()

    ax1.plot(t_c_1,Resta_c,'-',lw=0.9,label='Calibracion s/ fondo')
    ax1.legend()
    ax1.grid()
    fig.supxlabel('t (s)')
    fig.supylabel('Amplitud (V)')
    plt.xlim(0,7/frec_c)
    #plt.show()
    plt.savefig('testeo'+str(k)+'.png',dpi=400,facecolor='w')
    '''
    A partir de aca trabajo con señal de calibracion con el fondo restado.
    Les aplico recorte() sobre para tener N periodos enteros'''
    t_c_2, v_r_c_2 , v_c_2, N_ciclos_c, figura_c_2 = recorte(t_c_1,v_r_c_1,Resta_c,frec_c,'calibracion')

    v_c_3,_, _, _, _, _ , _ , fig, _,fig4,fig5,fig6 = fourier_señales_2(t_c_2,v_c_2,v_r_c_2,delta_t_c[k],filtro=0.2,frec_limite=2*frec_c,name=fnames_c[k])
    
    t_f_c_2 , fem_campo_c_2 , v_c_4 , delta_t_final_2= promediado_ciclos(t_c_2,v_r_c_2,v_c_3,frec_c,N_ciclos_c)
 
    '''
    Integro los ciclos: calcula sumas acumuladas y convierte a campo y magnetizacion
    Cte que dimensionaliza al campo en A/m a partir de la calibracion
    realizada sobre la bobina del RF'''
    pendiente_HvsI = 43.18*79.77 
    ordenada_HvsI = 2.73*79.77  
    C_norm_campo=Idc[k]*pendiente_HvsI+ordenada_HvsI
    
    #Susceptibilidad del patrón de calibración
    rho_bulk_Gd2O3 = 7.41e3      #Densidad del Gd2O3 bulk  [kg/m^3]
    rho_patron_Gd2O3 = 1748.6  # [kg/m^3] (Actualizado a portamuestra torneado)
    xi_bulk_Gd2O3_masa = (1.35e-4)*4*np.pi*1e-3  #[emu*m/g/A] = [m^3/kg]
    xi_patron_vol = xi_bulk_Gd2O3_masa*rho_patron_Gd2O3

    #Integral de la fem inducida, es proporcional a
    campo_ua0_c = delta_t_final_2*cumulative_trapezoid(fem_campo_c_2,initial=0)
    campo_ua_c = campo_ua0_c - np.mean(campo_ua0_c) #Campo en volt*segundo, falta llevar a ampere/metro.
    campo_c  = (campo_ua_c/max(campo_ua_c))*C_norm_campo#Doy unidades al campo 
    ciclos_H.append(campo_c)
    
    #Integral de la fem inducida, es proporcional a
    #la magnetizacion mas una constante
    mag_ua0_c = delta_t_final_2*cumulative_trapezoid(v_c_4,initial=0)
    mag_ua_c = mag_ua0_c-np.mean(mag_ua0_c)
    #Ajuste lineal sobre el ciclo del paramagneto
    pendiente , ordenada = np.polyfit(campo_c,mag_ua_c,1)
    polaridad_resta = np.sign(pendiente) 
    pendiente = pendiente*polaridad_resta
    mag_ua_c = mag_ua_c*polaridad_resta 
    pendientes.append(pendiente)
    ordenadas.append(ordenada)
    print(f'Pendiente: {pendiente:.2e}\nOrdenada: {ordenada:.2e}\n')
    #Calibración para pasar la magnetización a A/m
    calibracion_resta=xi_patron_vol/pendiente
    #Doy unidades a la magnetizacion de calibracion, ie, al paramagneto
    mag_c = calibracion_resta*mag_ua_c
    ciclos_M_ua.append(mag_ua_c)
    ciclos_M.append(mag_c)

    fig , ax =plt.subplots()    
    ax.plot(t_f_c_2,campo_c/max(campo_c),label='Campo')
    ax.plot(t_f_c_2,mag_ua_c/max(mag_ua_c),label='Magnetización')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('t (s)')
    plt.title('Campo y magnetización normalizados del paramagneto\n'+ fnames_c[k][:-4])


    #h=max(campo_orig_c)*xi_patron_vol
    #x_aux=np.linspace(min(campo_orig_c),max(campo_orig_c),num=1000)
    #y_aux = np.linspace(-h,h,num=1000)
    
    fig, ax = plt.subplots()
    ax.plot(campo_c,mag_c,'-',label=fnames_c[k][:-4])
    ax.plot(campo_c, (ordenada + pendiente*campo_c)*calibracion_resta,'--',lw=0.9,label=f'Ajuste Lineal: m = {pendiente:.2e} n = {ordenada:.2e} A/m')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('H (A/m)')
    plt.ylabel('M (A/m)')
    plt.title('Ciclo del paramagneto')
    #plt.savefig('Ciclo_histeresis_'+fnames_c[k]+'.png',dpi=300,facecolor='w')
    

#% Estadistica sobre pendientes y ordenadas
pend = ufloat(np.mean(pendientes),np.std(pendientes))
ord = ufloat(np.mean(ordenadas),np.std(ordenadas))
print(f'Pendiente {pend:.2e} V*s*m/A\nOrdenada {ord:.2e} V*s')

#grafico ciclos, rectas y estadistica de pend,ord

fig = plt.figure(figsize=(10,8),constrained_layout=True)
ax = fig.add_subplot(1,1,1)
for i in range(len(fnames_c)):
    h_min = -max(max(ciclos_H[i]),abs(min(ciclos_H[i])))      
    h_max = max(max(ciclos_H[i]),abs(min(ciclos_H[i])))
    h = np.linspace(h_min,h_max,1000)
    #plt.plot(ciclos_H[i],ciclos_M[i],label=f'{fnames_c[i][:-4]}')
    plt.plot(ciclos_H[i],ciclos_M_ua[i],label=f'{fnames_c[i][:-4]}')
    plt.plot(h, (ordenadas[i] + pendientes[i]*h),'--',label='AL')
plt.legend(loc='best',fancybox=True,ncol=1)
plt.grid()
plt.xlabel('H ($A/m$)',fontsize=15)
plt.ylabel('M $(V\cdot s)$',fontsize=15)
#plt.title('test',loc='left',y=0,fontsize=13)
plt.suptitle('Ciclos del paramagneto',fontsize=30)    
plt.text(0.7,0.05,f'Pendiente de calibracion promedio:\n {pend:^.2ue} $V\cdot s \cdot m/A$',bbox=dict(color='tab:orange',alpha=0.8),transform=ax.transAxes)
plt.savefig('Ciclos_paramagneto.png',dpi=300, facecolor='w')
# %% 

fig = plt.figure(figsize=(10,8),constrained_layout=True)
ax = fig.add_subplot(1,1,1)
for i in range(len(fnames_c)):      
    plt.plot(ciclos_H[i], (ordenadas[i] + pendientes[i]*campo_c)*calibracion_resta,label=f'{fnames_c[i][:-4]}')
ax.set_ylabel('M $(V\cdot s)$')
ax.set_xlabel('H $(A/m)$')
    #plot(Ciclos_eje_H_cal[i], Ciclos_eje_M_cal[i],c='r')

plt.legend(loc='best',fancybox=True,ncol=2)
plt.grid()
plt.xlabel('Campo (A/m)',fontsize=15)
plt.ylabel('Magnetización (A/m)',fontsize=15)
#plt.title('test',loc='left',y=0,fontsize=13)
plt.suptitle('Ajustes del paramagneto',fontsize=30)    

#plt.savefig('Ciclos_paramagneto.png',dpi=300, facecolor='w')
# %%
pend = ufloat(np.mean(pendientes),np.std(pendientes))
ord = ufloat(np.mean(ordenadas),np.std(ordenadas))
print(f'Pendiente {pend:.2e} V*s*m/A\nOrdenada {ord:.2e} V*s')
# %%
