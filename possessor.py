#%% 
# -*- coding: utf-8 -*-
"""
possessor.py 
(https://gost1980s.bandcamp.com/)

@author: Giuliano Andrés Basso
Basado en Planet_caravan_20210419.py 

Descripcion:
Toma uno o varios archivos con formato de nombre:
        'xxxkHz_yyydA_zzzMss_nombre.txt' para un solo archivo

        'xxxkHz_yyydA_zzzMss_nombre*.txt' para varios, con * cualquier caracter.

  xxx: frecuencia en kHz del campo.
  yyy: valor de la corriente Idc (Internal Direct Current) en deciAmpere (1e-1 A). 
  zzz: valor de muestreo (i.e., cuantos puntos por segundo registrados), en 
  megamuestras por segundo.

El valor de corriente es empleado para la amplitud del campo. 
El valor del muestreo es usado para la base temporal y el de 
frecuencia para verificar esa base temporal. 

Ademas el programa busca los archivos:
    'xxxkHz_yyydA_zzzMss_nombre_fondo.txt'
    'xxxkHz_yyydA_zzzMss_nombre_cal.txt'

Se ajustan funciones senoidales a las tres señales de referencia. 
Se elige el tiempo cero para que las tres referencias comiencen en fase. 
Se verifica que las frecuencias coincidan entre sí, y con el nombre del
archivo.

Se resta señal de muestra y señal de fondo, y señal de calibración y señal 
de fondo correspondiente. Para eso se dilata el tiempo del fondo 
multiplicándolo por el cociente de frecuencias (fondo/muestra, y 
fondo/calibración en cada caso) por si hay alguna pequeña discrepancia, 
tanto para la muestra como para el paramagneto de calibración. 
A partir de ahí se trabaja únicamente con estas restas y con las referencias 
de muestra y de calibración que contienen la información del campo.

Las señales se levantan crudas, y se omiten tanto el primer medio período como el último. 
De esta manera se evita el ruido que no pudo ser tratado adecuadamente al principio y al
final de cada medida.

A diferencia de los protocolos anteriores en Matlab, aca en lugar de los filtros que 
se solian usar (en privamera OWON por ej), hacemos directamente una transformada 
de Fourier y reconstruimos la señal a partir de sus armonicos impares. 
Se muestra el analisis espectral de la muestra asi como tambien la señal
reconstruida sobre la original para comparar. 

Se promedia resta y referencia sobre todos los períodos y se integra.

Se lleva el campo a unidades de A/m normalizando y multiplicando por el
campo máximo medido para el valor correspondiente de IDC. Se grafica el
ciclo en unidades de V*s para la calibración y se ajusta una recta. Con
la pendiente obtenida se lleva a unidades de A/m el eje de magnetización
para la muestra.

Se guarda un archivo con la imagen del gráfico del ciclo de histéresis,
y otro con los puntos del ciclo en ascii.

Se calculan la coercitividad y la remanencia.

Se calcula el valor de SAR integrando los ciclos.

Se imprime en un archivo de salida la siguiente informacion:
            Nombre del archivo
            Frecuencia (kHz)
            Campo máximo (kA/m)
            SAR (W/g)
            Coercitividad (kA/m)
            Magnetización Remanente (kA/m)
            Peor quita de ruido porcentual
"""
print(__doc__)
#%% Packages
'''Packages'''
import time
from datetime import datetime
from numpy.core.numeric import indices 
start_time = time.time() 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sc
from uncertainties import ufloat, unumpy
from scipy.signal import find_peaks 
from scipy.integrate import cumulative_trapezoid, trapezoid

from scipy.fft import fft, ifft, rfftfreq,irfft 
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from sklearn.metrics import r2_score
from pprint import pprint
#%% Funciones que usa el script
'''Funcion: fft_smooth()'''

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
     
'''Funcion: medida_cruda(path,delta_t).'''
def medida_cruda(path,delta_t):
    '''
    Funcion para levantar los archivos .txt en df de pandas.\n
    Identifica las dos señales de cada canal como señal y referencia.\n
    Recibe datos en mV y pasa a V. 
    '''
    df = pd.read_table(path,skiprows=3,names=['idx','v','v_r'],
                                    engine='python',sep='\s+')
    #paso de mV a V
    df['v'] = df['v']*0.001   
    df['v_r'] = df['v_r']*0.001 
    #agrego col de tiempos, la hago arrancar de 0 
    df.insert(loc=1,column='t',value=delta_t*(df['idx']-df['idx'][0]))
    #elimino columna de indice
    del df['idx'] 
    return df

'''Funcion: ajusta_seno(). Utiliza subrutina sinusoide()'''    
def sinusoide(t,A,B,C,D):
    '''
    Crea sinusoide con params: 
        A=offset, B=amp, C=frec, D=fase
    '''
    return(A + B*np.sin(2*np.pi*C*t - D))

from scipy.optimize import curve_fit
def ajusta_seno(t,v_r):
    '''
    Calcula params de iniciacion y ajusta sinusoide via curve_fit
    Para calacular la frecuencia mide tiempo entre picos
    '''
    offset0 = v_r.mean() 
    amplitud0=(v_r.max()-v_r.min())/2

    v_r_suave = fft_smooth(v_r,np.around(int(len(v_r)*6/1000)))
    indices, _ = find_peaks(v_r_suave,height=0)
    t_entre_max = np.mean(np.diff(t[indices]))
    frecuencia0 = 1 /t_entre_max
    
    fase0 = 2*np.pi*frecuencia0*t[indices[0]] - np.pi/2

    p0 = [offset0,amplitud0,frecuencia0,fase0]
    
    coeficientes, _ = curve_fit(sinusoide,t,v_r,p0=p0)
    
    offset=coeficientes[0]
    amplitud = coeficientes[1]
    frecuencia = coeficientes[2]
    fase = coeficientes[3]

    return offset, amplitud , frecuencia, fase

'''Funcion: resta_inter() '''
def resta_inter(t,v,v_r,fase,frec,offset,t_f,v_f,v_r_f,fase_f,frec_f,graf):    
    '''
    Funcion para la interpolacion y resta. 
    Grafica si parametro graf = graficos[1] = 1
    Desplazamiento temporal para poner en fase las referencias, y
    resta de valores medios de referencias.
    '''    
    '''
    Calculo el modulo 2 pi de las fases y calcula el tiempo de fase 0 
    '''
    t_fase = np.mod(fase,2*np.pi)/(2*np.pi*frec)
    t_fase_f = np.mod(fase_f,2*np.pi)/(2*np.pi*frec_f)
    '''
    Desplaza en tiempo para que haya coincidencia de fase e/ referencias.
    La amplitud debería ser positiva siempre por el valor inicial 
    del parámetro de ajuste.
    '''
    t_1 = t - t_fase 
    t_f_aux = t_f - t_fase_f     
    '''
    Correccion por posible diferencia de frecuencias dilatando el 
    tiempo del fondo 
    '''
    t_f_mod = t_f_aux*frec_f/frec 
    '''
    Resta el offset de la referencia
    '''
    v_r_1 = v_r - offset
    '''
    Resta medida y fondo interpolando para que corresponda mejor el
    tiempo. No trabaja más con la medidas individuales, sólo con la
    resta. Se toma la precaución para los casos de trigger distinto
    entre fondo y medida. Comienza en fase 0 o sea t=0.
    '''
    t_min=0
    t_max=t_f_mod.iloc[-1]
    '''
    Recorta el tiempo a mirar
    '''
    t_aux = t_1[np.nonzero((t_1>=t_min) & (t_1<=t_max))]
    ''' 
    Interpola el fondo a ese tiempo
    '''
    interpolacion_aux = np.interp(t_aux,t_f_mod,v_f)
    interpolacion = np.empty_like(v)   
    '''
    Cambia índice viejo por índice nuevo en la base original
    '''
    for w in range(0,len(t_1),1):  
        #obtengo el indice donde esta el minimo de la resta: 
        index_min_m = np.argmin(abs(t_aux - t_1[w]))
        #defino c/ elemento de interpolacion: 
        interpolacion[w] = interpolacion_aux[index_min_m]
    '''
    Defino la resta entre la señal (m o c) y la interpolacion de la señal
    de fondo'''
    Resta = v - interpolacion
    '''
    Comparacion de las referencias de muestra y fondo desplazadas en
    tiempo y offset
    '''
    
    if graf!=0:
        '''Rutina de ploteo para resta_inter'''
        def ploteo(t_f_mod,v_r_f,t_1,v_r_1):
            fig = plt.figure(figsize=(10,8),constrained_layout=True)
            ax = fig.add_subplot(2,1,1)
            plt.plot(t_1,v_r_1,lw=1,label=str(graf).capitalize())
            plt.plot(t_f_mod,v_r_f,lw=1,label='Fondo')
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('t (s)')
            plt.ylabel('Referencia (V)')
            plt.title('Referencias desplazadas y restados sus offsets',loc='left')
            
            ax = fig.add_subplot(2,1,2)
            plt.plot(t_f_mod,v_f,lw=1,label='Fondo')
            plt.plot(t_1,v,lw=1,label=str(graf).capitalize())
            plt.plot(t_1,interpolacion,lw=1,label='Interpolacion del fondo al tiempo de la '+str(graf))
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('t (s)')
            plt.ylabel('Señal (V)')
            plt.title(str(graf).capitalize()+' y fondo',loc='left')
            #plt.savefig('Comparacion_de_medidas_'+str(graf)+'.png',dpi=300)
            fig.suptitle('Comparacion de señales',fontsize=20)
            return fig
        figura = ploteo(t_f_mod,v_r_f,t_1,v_r_1)
    else:
        figura = 'Figura off'

    return Resta , t_1 , v_r_1 , figura 

'''Funcion: encuentra_ruido().Es subrutina del filtro Actis, en filtrando_ruido()'''
def encuentra_ruido(t,v,ancho,entorno):
    from scipy.signal import lfilter
    '''
    Toma una señal (t,v) y calcula la derivada de v respecto de t.
    Valor absoluto medio: ruido_tranqui. 
    Marca los puntos con derivada en valor absoluto mayor 
    que "ancho" veces "ruido_tranqui" y un entorno de estos puntos 
    igual a "entorno" puntos para cada lado.
    '''
    '''Suaviza con un promedio leve'''
    WindowSize = 5
    be = (1/WindowSize)*np.ones(WindowSize)
    t_1 = t[WindowSize+1:]-WindowSize*(t[1]-t[0])/2
    v_fe = lfilter(be,1,v)
    v_1 = v_fe[WindowSize+1:] 

    '''Calcula la derivada de v respecto a t'''
    derivada = np.diff(v_1)/np.diff(t_1)
    t_2 = t_1[:-1]+(t_1[1]-t_1[0])/2

    '''Suaviza la derivada'''
    t_3 = t_2[WindowSize+1:] - WindowSize*(t_2[1]-t_2[0])/2
    derivada0 = lfilter(be,1,derivada)
    derivada2 = derivada0[WindowSize+1:]

    '''
    El ruido caracteristico de la señal es el valor medio 
    del valor absoluto de la derivada
    '''
    ruido_tranqui = np.mean(abs(derivada2))
    aux_1 = np.zeros(len(derivada2)+1)
    '''
    Marca puntos que superan en ancho veces el ruido normal
    '''
    for jq in range(len(derivada2)):
        if abs(derivada2[jq])>ancho*ruido_tranqui:
            aux_1[jq] = 1
        else:
            aux_1[jq] = 0    
    '''Prepara el marcador '''
    marcador = np.zeros_like(derivada2)
    '''
    Si hay un solo cambio de signo en la derivada, 
    no lo marca, ya que debe tratarse de un pico 
    natural de la señal
    ''' 
    for jq in range(entorno,len(derivada2)-entorno):
        if max(aux_1[jq-entorno:jq+entorno]) == 1:
            marcador[jq + int(np.round(entorno/2))] = 1
        else:
            marcador[jq + int(np.round(entorno/2))] = 0
    
    '''Acomodo los extremos '''
    for jq in range(entorno):
        if marcador[entorno+1] == 1:
            marcador[jq] = 1
        if marcador[len(derivada2)- entorno] == 1:
            marcador[len(derivada2)-jq-1]=1
        
    return t_3, marcador

'''Funcion: filtrando_ruido()'''
def filtrando_ruido(t,v_r,v,filtrar,graf):
    '''
    Sin filtrar: filtrarmuestra/cal = 0
    Actis:       filtrarmuestra/cal = 1
    Fourier:     filtrarmuestra/cal = 2   
    Fourier+Actis: filtrarmuestra/cal = 3   
    ''' 
    if filtrar == 0:
        t_2 = t
        v_r_2 = v_r
        v_2 = v
        figura_2 = 'No se aplico filtrado'

    elif filtrar==2 or filtrar==3: 
        '''Filtro por Fourier'''
        freq = np.around(len(v_r)/5)
        v_2 = fft_smooth(v,freq)
        v_r_2 = fft_smooth(v_r,freq)
        t_2 = t   

        if graf !=0:
            '''Control de que el suavizado final sea satisfactorio'''
            figura_2 = plt.figure(figsize=(10,8),constrained_layout=True)
            ax1 = figura_2.add_subplot(2,1,1)
            plt.plot(t,v_r,'.-',label='Sin filtrar')
            plt.plot(t_2,v_r_2,lw=1,label='Filtrada')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.title('Señal de referencia',loc='left',fontsize=15)
            plt.ylim(1.25*min(v_r_2),1.25*max(v_r_2))            
            ax2 = figura_2.add_subplot(2,1,2,sharex=ax1)
            plt.plot(t,v,'.-',label='Sin filtrar')
            plt.plot(t_2,v_2,lw=1,label='Filtrada')
            #plt.plot(t,v,lw=1,label='Zona de ruido')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.xlabel('t (s)')
            plt.title('Señal de '+ str(graf),loc='left',fontsize=15)  
            plt.xlim(t[0],t[-1]/4)#provisorio
            figura_2.suptitle('Filtro de Fourier - '+str(graf).capitalize(),fontsize=20)
            
        else:
            figura_2 = 'Figura off'   

    elif filtrar ==1: #filtro Actis
        '''
        Identifica el ruido: factor del ruido natural a partir del 
        cual se considera que hay que filtrar
        '''
        ancho=2.5
        '''
        Puntos a ambos lados del sector de señal ruidosa que serán
        incluidos en el ruido.
        '''
        entorno=5

        '''Aca ejecuto funcion encuentra_ruido(t,v,ancho,enterno)
        obtengo: t_2 y marcador'''
        t_2 , marcador = encuentra_ruido(t,v,ancho,entorno)

        '''
        Ajuste ruido
        Params: 
            ancho= puntos a cada lado de la region a filtrar que serán considerados para el ajuste
            grado_pol= grado del polinomio a ajustar.
        '''
        puntos_ajuste=80
        grado_pol=3
        '''Tiempos y señales en las mismas dimensiones que los marcadores'''
        interpolador = sc.interpolate.interp1d(t,v,kind='slinear') 
        v_2 = interpolador(t_2)

        interpolador_r = sc.interpolate.interp1d(t,v_r,kind='slinear') 
        v_r_2 = interpolador_r(t_2)

        '''
        Comienza a filtrar una vez que tiene suficientes puntos detras
        '''
        w=puntos_ajuste + 1
        '''
        Barre la señal. NO FILTRA ni el principio ni el final, 
        por eso mas adelante se eliminan 1er y ultimo semiperiodos.
        ''' 
        while w<len(v_2):
            
            if marcador[w-1]==0:
                '''Si no hay ruido deja la señal como estaba'''
                w += 1    
            elif marcador[w-1]==1:
                '''Si hay ruido'''
                q=w
                '''Busca hasta donde llega el ruido'''
                while marcador[q-1]==1 and q<len(v_2):
                    q+=1
                '''si se tienen suficientes puntos del otro lado
                realiza el ajuste'''
                if q<len(v_2)-puntos_ajuste:
                    y = np.concatenate((v_2[w-puntos_ajuste:w],v_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    x = np.concatenate((t_2[w-puntos_ajuste:w],t_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    p = np.polyfit(x,y,grado_pol)
                    v_2[w:q-1]= np.polyval(p,t_2[w:q-1])

                    y_r = np.concatenate((v_r_2[w-puntos_ajuste:w],v_r_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    x_r = np.concatenate((t_2[w-puntos_ajuste:w],t_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    p_r= np.polyfit(x_r,y_r,grado_pol)
                    v_r_2[w:q-1]= np.polyval(p_r,t_2[w:q-1])
                w=q 

        if graf !=0:
            '''Control de que el suavizado final sea satisfactorio'''
            figura_2 = plt.figure(figsize=(10,8),constrained_layout=True)
            ax1 = figura_2.add_subplot(2,1,1)
            plt.plot(t,v_r,'.-',label='Referencia de '+ str(graf))
            plt.plot(t_2,v_r_2,lw=1,label='Referencia de '+str(graf)+' filtrada')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.ylim(1.25*min(v_r_2),1.25*max(v_r_2))
            plt.title('Señal de referencia',loc='left',fontsize=15)
            
            ax2 = figura_2.add_subplot(2,1,2,sharex=ax1)
            plt.plot(t,v,'-',lw=1.5,label='Resta de señales')
            plt.plot(t_2,v_2,lw=0.9,label='Sin ruido')
            plt.plot(t_2,marcador,lw=1, alpha=0.8 ,label='Zona de ruido')
            #plt.plot(t,v,lw=1,label='Zona de ruido')
            plt.legend(ncol=3,loc='lower center')
            plt.grid()
            plt.xlabel('t (s)')
            plt.title('Señal de ' + str(graf),loc='left',fontsize=15)
            plt.xlim(t[0],t[-1]/4)#provisorio
            figura_2.suptitle('Filtro Actis - '+str(graf).capitalize(),fontsize=20)
            
        else:
            figura_2 = "Figura Off"
        
    return t_2 , v_r_2 , v_2 ,figura_2 

'''Funcion: recorte()'''
def recorte(t,v_r,v,frecuencia,graf):
    '''
    Recorta un numero entero de periodos o ciclos,arrancando en fase 0 (campo max o campo min segun polaridad)
    Grafico: señal de muestra/calibracion, s/ fondo, s/ valor medio y recortadas a un numero entero de ciclos.
    '''
    #Numero de ciclos
    N_ciclos =  int(np.floor(t[-1]*frecuencia)) 
    
    #Indices ciclo
    indices_recorte = np.nonzero(np.logical_and(t>=0,t<N_ciclos/frecuencia))  
    
    #Si quisiera recortar ciclos a ambos lados
    # largo = indices_ciclo[-1][0]
    #if np.mod(largo,N_ciclos) == 0:
    # largo = largo - np.mod(largo,N_ciclos)
    #elif np.mod(largo,N_ciclos) <= 0.5:
        #largo = largo - np.mod(largo,N_ciclos)
    #else:
    # largo = largo + N_ciclos - np.mod(largo,N_ciclos)
    '''
    Recorto los vectores
    '''
    t_2 = t[indices_recorte]
    v_2 = v[indices_recorte]
    v_r_2 = v_r[indices_recorte]
    if graf !=0:
        '''
        Señal de muestra/calibracion, s/ fondo, s/ valor medio y 
        recortadas a un numero entero de ciclos
        '''
        figura = plt.figure(figsize=(10,8),constrained_layout=True)
        ax1 = figura.add_subplot(2,1,1)
        plt.plot(t_2,v_r_2,'.-',lw=1)
        plt.grid()
        plt.title('Señal de referencia',loc='left',fontsize=15)
        plt.ylabel('Señal (V)')
        plt.axvspan(0,1/frecuencia, facecolor='#2ca02c',label='Período 1/{}'.format(N_ciclos),alpha=0.4)
        plt.ylim(1.3*min(v_r_2),1.3*max(v_r_2))            
        plt.legend(loc='lower left')
        ax2 = figura.add_subplot(2,1,2,sharex=ax1)
        plt.plot(t_2,v_2,'.-',lw=1)
        plt.axvspan(0,1/frecuencia, facecolor='#2ca02c',label='Período 1/{}'.format(N_ciclos),alpha=0.4)
        plt.legend(loc='lower left')
        plt.grid()
        plt.xlabel('t (s)')
        plt.ylabel('Señal (V)')
        plt.ylim(1.3*min(v_2),1.3*max(v_2))
        plt.title('Señal de '+ str(graf),loc='left',fontsize=15)  
        figura.suptitle('Número entero de períodos - '+str(graf).capitalize(),fontsize=20)
    else:
        figura ='Figura off'

    return t_2 , v_r_2 , v_2 , N_ciclos, figura

'''Funcion: promediado_ciclos()'''
def promediado_ciclos(t,v_r,v,frecuencia,N_ciclos):
    '''
    '''
    t_f = t[t<t[0]+1/frecuencia]
    v_r_f = np.zeros_like(t_f)
    v_f = np.zeros_like(t_f)
    for indx in range(N_ciclos):
        if t_f[-1] + indx/frecuencia < t[-1]: 
            interpolador_r = sc.interpolate.interp1d(t,v_r,kind='linear')
            interpolador = sc.interpolate.interp1d(t,v,kind='linear')
            v_r_f = v_r_f + interpolador_r(t_f + indx/frecuencia)/N_ciclos
            v_f = v_f + interpolador(t_f + indx/frecuencia)/N_ciclos

        else: #True en la ultima iteracion
            interpolador_r_2 = sc.interpolate.interp1d(t,v_r,kind='slinear')
            interpolador_2 = sc.interpolate.interp1d(t,v,kind='slinear')
            v_r_f = v_r_f + interpolador_r_2(t_f + (indx-1)/frecuencia)/N_ciclos
            v_f = v_f + interpolador_2(t_f + (indx-1)/frecuencia)/N_ciclos
    
    '''Quita valor medio'''
    v_f = v_f - v_f.mean()
    v_r_f = v_r_f - v_r_f.mean()
    '''Paso temporal'''
    delta_t = (t_f[-1]-t_f[0])/len(t_f)
    return t_f , v_r_f, v_f , delta_t
#%% El kernel
def fourier_señales(t,t_c,v,v_c,v_r_m,v_r_c,delta_t,polaridad,filtro,frec_limite_m,frec_limite_cal,name):
    '''
    Toma señales de muestra, calibracion y referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos.
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se conoce la polaridad de la señal(del ajuste lineal sobre el ciclo paramagnetico). 
    '''
    t = t - t[0] #Muestra 
    t_r = t.copy() #y su referencia
    t_c = t_c - t_c[0] #Calibracion 
    t_r_c = t_c.copy() #y su referencia
    
    y = polaridad*v     #muestra (magnetizacion)
    y_c = polaridad*v_c #calibracion (magnetizacion del paramagneto)
    y_r = v_r_m        #referencia muestra (campo)
    y_r_c = v_r_c      #referencia calibracion (campo)
    
    N = len(v)
    N_c = len(v_c)
    N_r_m = len(v_r_m)
    N_r_c = len(v_r_c)
    
    #Para que el largo de los vectores coincida
    if len(t)<len(y): #alargo t
        t = np.pad(t,(0,delta_t*(len(y)-len(t))),mode='linear_ramp',end_values=(0,max(t)+delta_t*(len(y)-len(t))))
    elif len(t)>len(y):#recorto t    
        t=np.resize(t,len(y))

    if len(t_c)<len(y_c): #alargo t
        t_c = np.pad(t_c,(0,delta_t*(len(y_c)-len(t_c))),mode='linear_ramp',end_values=(0,max(t_c)+delta_t*(len(y_c)-len(t_c))))
    elif len(t_c)>len(y_c):#recorto t    
        t_c=np.resize(t_c,len(y_c))
    
    #Idem referencias
    if len(t_r)<len(y_r): #alargo t
        t_r = np.pad(t_r,(0,delta_t*(len(y_r)-len(t_r))),mode='linear_ramp',end_values=(0,max(t_r)+delta_t*(len(y_r)-len(t_r))))
    elif len(t_r)>len(y_r):#recorto t    
        t_r=np.resize(t_r,len(y_r))

    if len(t_r_c)<len(y_r_c): #alargo t
        t_r_c = np.pad(t_r_c,(0,delta_t*(len(y_r_c)-len(t_r_c))),mode='linear_ramp',end_values=(0,max(t_r_c)+delta_t*(len(y_r_c)-len(t_r_c))))
    elif len(t_r_c)>len(y_r_c):#recorto t    
        t_r_c=np.resize(t_r_c,len(y_r_c))

#Aplico transformada de Fourier
    f = rfftfreq(N,d=delta_t) #obtengo la mitad de los puntos, porque uso rfft
    f_HF = f.copy() 
    #f_HF = f_HF[np.nonzero(f>=frec_limite)] #aca estan el resto 
    f = f[np.nonzero(f<=frec_limite_m)] #limito frecuencias 
    g_aux = fft(y,norm='forward') 
    #“forward” applies the 1/n factor on the forward tranform
    g = abs(g_aux)  #magnitud    
    fase = np.angle(g_aux)
    
    #Idem p/ calibracion
    f_c = rfftfreq(N_c, d=delta_t)
    f_c_HF = f_c.copy()
    f_c = f_c[np.nonzero(f_c<=frec_limite_cal)]
    g_c_aux = fft(y_c,norm='forward') 
    g_c = abs(g_c_aux)
    fase_c= np.angle(g_c_aux)
    
    #Idem p/ Referencia
    f_r = rfftfreq(N_r_m, d=delta_t)
    f_r = f_r[np.nonzero(f_r<=frec_limite_m)]
    g_r_aux = fft(y_r,norm='forward')
    g_r = abs(g_r_aux)
    fase_r = np.angle(g_r_aux)
    #y para ref de calibracion
    f_r_c = rfftfreq(N_r_c, d=delta_t)
    f_r_c = f_r_c[np.nonzero(f_r_c<=frec_limite_cal)]
    g_r_c_aux = fft(y_r_c,norm='forward')
    g_r_c = abs(g_r_c_aux)
    fase_r_c = np.angle(g_r_c_aux)
     
    #Recorto vectores hasta frec_limite/_cal
    g_HF = g.copy() 
    g_c_HF = g_c.copy()
    #g_HF = g_HF[np.nonzero(f>=frec_limite)]#magnitud de HF
    g = np.resize(g,len(f))
    g_c = np.resize(g_c,len(f_c))
    g_r = np.resize(g_r,len(f_r))
    g_r_c = np.resize(g_r_c,len(f_r_c))
    g_HF = np.resize(g_HF,len(f_HF))
    g_HF[np.argwhere(f_HF<=frec_limite_m)]=0 #Anulo LF
    g_c_HF = np.resize(g_c_HF,len(f_c_HF))
    g_c_HF[np.argwhere(f_c_HF<=frec_limite_cal)]=0 #Anulo LF

#Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    indices,_=find_peaks(abs(g),threshold=max(g)*filtro)
    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)
    
    indices_c,_=find_peaks(abs(g_c),height=max(g_c)*filtro)

    indices_r,_=find_peaks(abs(g_r),height=max(g_r)*filtro)

    indices_r_c,_=find_peaks(abs(g_r_c),height=max(g_r_c)*filtro)

    #En caso de frecuencia anomala menor que la fundamental en Muestra
    for elem in indices:
        if f[elem]<0.95*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[elem]))
            indices = np.delete(indices,0)
    #En caso de frecuencia anomala menor que la fundamental en Calibracion
    for elem in indices_c:
        if f_c[elem]<0.95*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de calibracion {:.2f} Hz\n'.format(f_c[elem]))
            indices_c = np.delete(indices_c,0)
            
    armonicos = f[indices]
    amplitudes = g[indices]
    fases = fase[indices]

    armonicos_c = f_c[indices_c]
    amplitudes_c = g_c[indices_c]
    fases_c = fase_c[indices_c]

    armonicos_r = f_r[indices_r]
    amplitudes_r = g_r[indices_r]
    fases_r = fase_r[indices_r]

    armonicos_r_c = f_r_c[indices_r_c]
    amplitudes_r_c = g_r_c[indices_r_c]
    fases_r_c = fase_r_c[indices_r_c]
    #Imprimo tabla 
    print(f'{k+1}/{len(fnames_m)}\nArchivo: {fnames_m[k]:s}')
    print('''Espectro de la señal de referencia:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_r)):
        print(f'{armonicos_r[i]:<10.2f}    {amplitudes_r[i]/max(amplitudes_r):>12.2f}    {fases_r[i]:>12.4f}')
    
    print('''\nEspectro de la señal de calibracion:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_c)):
        print(f'{armonicos_c[i]:<10.2f}    {amplitudes_c[i]/max(amplitudes_c):>12.2f}    {fases_c[i]:>12.4f}')
    
    print('''\nEspectro de la señal de muestra:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices)):
        print(f'{armonicos[i]:<10.2f}    {amplitudes[i]/max(amplitudes):>12.2f}    {fases[i]:>12.4f}')
  
#Frecuencias/indices multiplo impar/par de la fundamental
    frec_multip = []
    indx_impar = []
    indx_par=[]

    for n in range(int(frec_limite_m//int(armonicos[0]))):
        frec_multip.append((2*n+1)*armonicos[0]/1000)
        if (2*n+1)*indices[0]<=len(f):
            indx_impar.append((2*n+1)*indices[0])
            indx_par.append((2*n)*indices[0])
    
    frec_multip_c = []
    indx_impar_c = []
    indx_par_c=[]
    for n in range(int(frec_limite_cal//int(armonicos_c[0]))):
        frec_multip_c.append((2*n+1)*armonicos_c[0]/1000)
        if (2*n+1)*indices_c[0]<=len(f_c):
            indx_impar_c.append((2*n+1)*indices_c[0])
            indx_par_c.append((2*n)*indices_c[0])

    f_impar= f[indx_impar] #para grafico 1.0
    amp_impar= g[indx_impar]
    fases_impar= fase[indx_impar]
    del indx_par[0]
    f_par= f[indx_par] 
    amp_par= g[indx_par]
    fases_par= fase[indx_par]

    f_impar_c= f_c[indx_impar_c] #para grafico 1.1
    amp_impar_c= g_c[indx_impar_c]
    fases_impar_c= fase_c[indx_impar_c]
    del indx_par_c[0]
    f_par_c= f_c[indx_par_c] 
    amp_par_c= g_c[indx_par_c]
    fases_par_c= fase_c[indx_par_c]

#Reconstruyo señal impar con ifft p/ muestra
    h_aux_impar = np.zeros(len(f),dtype=np.cdouble)
    for W in indx_impar:
        h_aux_impar[W]=g_aux[W]
    rec_impares = irfft(h_aux_impar,n=len(t),norm='forward')
#Reconstruyo señal par con ifft
    h_aux_par = np.zeros(len(f),dtype=np.cdouble)
    for Z in indx_par:
        h_aux_par[Z]=g_aux[Z] 
    rec_pares = irfft(h_aux_par,n=len(t),norm='forward')

#Idem Calibracion
    h_c_aux_impar = np.zeros(len(f_c),dtype=np.cdouble)
    for W in indx_impar_c:
        h_c_aux_impar[W]=g_c_aux[W]
    rec_impares_c = irfft(h_c_aux_impar,n=len(t_c),norm='forward')
#Reconstruyo señal par con ifft
    h_c_aux_par = np.zeros(len(f_c),dtype=np.cdouble)
    for Z in indx_par_c:
        h_c_aux_par[Z]=g_c_aux[Z] 
    rec_pares_c = irfft(h_c_aux_par,n=len(t_c),norm='forward')

#Reconstruyo señal limitada con ifft
    #g_aux = np.resize(g_aux,len(f))
    #rec_limitada = irfft(g_aux,n=len(t),norm='forward')

#Reconstruyo señal de alta frecuencia
    rec_HF = irfft(g_HF,n=len(t),norm='forward')
    rec_c_HF = irfft(g_c_HF,n=len(t_c),norm='forward')
#Resto HF a la señal original y comparo con reconstruida impar
    resta = y - rec_HF
#Veo que tanto se parecen
    #r_2 = r2_score(rec_impares,rec_limitada)
    #r_2_resta  = r2_score(rec_impares,resta)
    
#Grafico 1.0 (Muestra): 
    fig = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Muestra',fontsize=20)
#Señal Orig + Ref
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t,y/max(y),'.-',lw=0.9,label='Muestra')
    ax1.plot(t_r,y_r/max(y_r),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title('Muestra y referencia - '+str(name), loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  
#Espectro de Frecuencias 
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(f/1000,g,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar/1000,amp_impar,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.axhline(y=max(g)*filtro,xmin=0,xmax=1,c='tab:orange',label=f'Filtro ({filtro*100}%)')
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite_m/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.legend(loc='best')
#  Espectro de Fases 
    ax3 = fig.add_subplot(3,1,3)
    ax3.vlines(armonicos/1000,ymin=0,ymax=fases)
    ax3.stem(armonicos/1000,fases,basefmt=' ')
    ax3.scatter(f_impar/1000,fases_impar,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    ax3.scatter(f_par/1000,fases_par,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

#Grafico 1.1 (Calibracion): 
    fig4 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Calibracion',fontsize=20)
#Señal Orig + Ref
    ax1 = fig4.add_subplot(3,1,1)
    ax1.plot(t_c,y_c/max(y_c),'.-',lw=0.9,label='Calibracion')
    ax1.plot(t_r_c,y_r_c/max(y_r_c),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title('Calibracion y referencia - '+str(name)+'_cal', loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  
#Espectro de Frecuencias 
    ax2 = fig4.add_subplot(3,1,2)
    ax2.plot(f_c/1000,g_c,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar_c/1000,amp_impar_c,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    #ax2.scatter(f_par_c/1000,amp_par_c,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite_cal/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f_c)/1000)
    ax2.legend(loc='best')
#  Espectro de Fases 
    ax3 = fig4.add_subplot(3,1,3)
    ax3.vlines(armonicos_c/1000,ymin=0,ymax=fases_c)
    ax3.stem(armonicos_c/1000,fases_c,basefmt=' ')
    ax3.scatter(f_impar_c/1000,fases_impar_c,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    #ax3.scatter(f_par_c/1000,fases_par_c,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip_c:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

#Redefino angulos p/ Fasorial impar
    r0 = 1
    theta_0 = 0
    r = amp_impar/max(amp_impar)
    defasaje_m =  fases_r-fases_impar
    r_c = amp_impar_c/max(amp_impar_c)
    defasaje_c = fases_r_c - fases_impar_c

#Grafico 2.0: Espectro Impar, Fasorial, Original+Rec_impar (Muestra)
    fig2 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar',fontsize=20)
# Señal Original + Reconstruida impar
    ax1=fig2.add_subplot(3,1,1)
    ax1.plot(t,y,'.-',lw=0.9,label='Señal original')
    #ax1.plot(t,reconstruida*max(rec2),'r-',lw=0.9,label='Reconstruida ({} armónicos)'.format(len(armonicos)))
    #ax1.plot(t,rec_limitada,'-',lw=0.9,label='Filtrada ({:.0f} kHz)'.format(frec_limite/1e3))
    ax1.plot(t,rec_impares,'-',lw=1.7,c='tab:orange',label='Componentes impares')
    ax1.plot(t,rec_pares,'-',lw=1.1,c='tab:green',label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title(str(name))
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax1.grid() 
    ax1.legend(loc='best')
# Espectro en fases impares
    ax2=fig2.add_subplot(3,1,2)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    ax2.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax2.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    #for item in f_impar:
        #ax2.axvline(item,0,1,c='tab:orange',lw=0.9)   
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de la señal reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
# inset
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar/1000,fases_impar,label='Fases')
    axin.vlines(f_impar/1000, ymin=0, ymax=fases_impar)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
# Fasorial impares
    ax3=fig2.add_subplot(3,1,3,polar=True)
    ax3.scatter(theta_0,r0,label='Referencia',marker='D',c='tab:red')
    ax3.plot([0,theta_0], [0,1],'-',c='tab:red')
    #ax3.plot(defasaje_m,r,'-',c='tab:blue',lw=0.7)
    ax3.scatter(defasaje_m, r, label = 'Muestra',c='tab:blue')
    for i in range(len(defasaje_m)):
        ax3.plot([0, defasaje_m[i]], [0, r[i]],'-o',c='tab:blue')
    ax3.spines['polar'].set_visible(False)  # Show or hide the plot spine
    ax3.set_rmax(1.1)
    ax3.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
    ##ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax3.grid(True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.legend(loc='upper left',bbox_to_anchor=(1.02,0.5,0.4,0.5))
    ax3.set_title('Retraso respecto al campo', loc='center', fontsize=13,va='bottom')

#Grafico 2.1: Espectro Impar, Fasorial, Original+Rec_impar (Calibracion)
    fig5 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar (calibracion)',fontsize=20)
# Señal Original + Reconstruida impar (Calibracion)
    ax1=fig5.add_subplot(3,1,1)
    ax1.plot(t_c,y_c,'.-',lw=0.9,label='Señal original')
    ax1.plot(t_c,rec_impares_c,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t_c,rec_pares_c,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax1.grid() 
    ax1.legend(loc='best')
# Espectro en fases impares (Calibracion)
    ax2=fig5.add_subplot(3,1,2)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    ax2.scatter(f_impar_c/1000,amp_impar_c,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar_c/1000, ymin=0, ymax=amp_impar_c)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax2.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    
    #for item in f_impar:
        #ax2.axvline(item,0,1,c='tab:orange',lw=0.9)   
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de la señal de calibracion reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
# inset (Calibracion)
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar_c/1000,fases_impar_c,label='Fases')
    axin.vlines(f_impar_c/1000, ymin=0, ymax=fases_impar_c)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
# Fasorial impares (Calibracion)
    ax3=fig5.add_subplot(3,1,3,polar=True)
    ax3.scatter(theta_0,r0,label='Referencia',marker='D',c='tab:red')
    ax3.plot([0,theta_0], [0,1],'-',c='tab:red')
    #ax3.plot(defasaje_m,r,'-',c='tab:blue',lw=0.7)
    ax3.scatter(defasaje_c, r_c, label = 'Muestra',c='tab:blue')
    for i in range(len(defasaje_c)):
        ax3.plot([0, defasaje_c[i]], [0, r_c[i]],'-o',c='tab:blue')
    ax3.spines['polar'].set_visible(False)  # Show or hide the plot spine
    ax3.set_rmax(1.1)
    ax3.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
    ##ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax3.grid(True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.legend(loc='upper left',bbox_to_anchor=(1.02,0.5,0.4,0.5))
    ax3.set_title('Retraso respecto al campo', loc='center', fontsize=13,va='bottom')

#Grafico 3: Altas frecuencias (muestra)
    fig3 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Altas frecuencias',fontsize=20)
# Espectro en fases impares
    ax1=fig3.add_subplot(2,1,1)
    ax1.stem(f_impar/1000,amp_impar,linefmt='C0-',basefmt=' ',markerfmt='.',bottom=0.0,label='armónicos impares')
    ax1.stem(f_par/1000,amp_par,linefmt='C2-',basefmt=' ',markerfmt='.g',bottom=0.0,label='armónicos pares')
    ax1.plot(f_HF[f_HF>frec_limite_m]/1000,g_HF[f_HF>frec_limite_m],'.-',lw=0.9,c='tab:orange',label='Alta frecuencia')
    #ax1.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    #ax1.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    #ax1.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    #ax1.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax1.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    ax1.legend(loc='best')
    ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    ax1.set_xlabel('Frecuencia (kHz)')
    ax1.set_ylabel('|F{$\epsilon$}|')   
    #ax1.set_xlim(0,max(f)/1000)
    #ax1.set_ylim(0,max(amp_impar)*1.1)
    ax1.grid()

# Señal HF + Reconstruida LF
    ax2=fig3.add_subplot(2,1,2)
    ax2.plot(t,rec_impares,'-',lw=1,label=f'Componentes impares (f<{frec_limite_m/1e6:.0f} MHz)',c='tab:blue',zorder=3)
    ax2.plot(t,rec_pares,'-',lw=1,label=f'Componentes pares (f<{frec_limite_m/1e6:.0f} MHz)',c='tab:green',zorder=3)  
    ax2.plot(t,rec_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite_m/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    ax2.plot(t,y,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1,alpha=0.8)    
    #ax2.plot(t,resta,'-',lw=0.9,label='Resta')  
    ax2.set_xlabel('t (s)')
    ax2.set_xlim(0,5/armonicos[0])
    ax2.set_ylim(1.1*min(y),1.1*max(y))
    ax2.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax2.set_title(str(name))
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax2.grid() 
    ax2.legend(loc='best')

#Grafico 3.1: Altas frecuencias (calibracion)
    fig6 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Altas frecuencias',fontsize=20)
# Espectro en fases impares
    ax1=fig6.add_subplot(2,1,1)
    ax1.stem(f_impar_c/1000,amp_impar_c,linefmt='C0-',basefmt=' ',markerfmt='.',bottom=0.0,label='armónicos impares')
    #ax1.stem(f_par_c/1000,amp_par_c,linefmt='C2-',basefmt=' ',markerfmt='.g',bottom=0.0,label='armónicos pares')
    ax1.plot(f_c_HF[f_c_HF>frec_limite_cal]/1000,g_c_HF[f_c_HF>frec_limite_cal],'.-',lw=0.9,c='tab:orange',label='Alta frecuencia')
    #ax1.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    #ax1.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    #ax1.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    #ax1.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax1.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    ax1.legend(loc='best')
    ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    ax1.set_xlabel('Frecuencia (kHz)')
    ax1.set_ylabel('|F{$\epsilon$}|')   
    #ax1.set_xlim(0,max(f)/1000)
    #ax1.set_ylim(0,max(amp_impar)*1.1)
    ax1.grid()

# Señal HF + Reconstruida LF
    ax2=fig6.add_subplot(2,1,2)
    ax2.plot(t_c,rec_impares_c,'-',lw=1,label=f'Componentes impares (f<{frec_limite_cal/1e6:.0f} MHz)',c='tab:blue',zorder=3)
    ax2.plot(t_c,rec_pares_c,'-',lw=1,label=f'Componentes pares (f<{frec_limite_cal/1e6:.0f} MHz)',c='tab:green',zorder=3)  
    ax2.plot(t_c,rec_c_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite_cal/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    ax2.plot(t_c,y_c,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1,alpha=0.8)    
    #ax2.plot(t,resta,'-',lw=0.9,label='Resta')  
    ax2.set_xlabel('t (s)')
    ax2.set_xlim(0,5/armonicos_c[0])
    ax2.set_ylim(1.1*min(y_c),1.1*max(y_c))
    ax2.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax2.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax2.grid() 
    ax2.legend(loc='best')

    return armonicos, armonicos_r, amplitudes, amplitudes_r, fases , fases_r , fig, fig2, indices, indx_impar, rec_impares,rec_impares_c,fig3,fig4,fig5,fig6

#%% Manual Settings
'''
Input necesario de parte del usuario y definiciones preliminares
'''
#¿Qué archivos desea abrir?

todos=0
        #todos = 0 => Selecciono los archivos
        #      = 1 => Selecciono carpeta/directorio y abro 
        # todos los archivos en la carpeta seleccionada
        # cuyo nombre de archivo termine con el nombre 
        # de muestra:
nombre='A0'
#Caso contrario no hace falta especificar nombre

ciclos_en_descongelamiento = 0
        # = 1 => 1 archivo fondo y cal p/ todo el analisis
        # = 0 => 1 f 1 cal x archivo muestra               

nombre_T='FG_T'  #para medidas en descongelamiento

#¿Qué gráficos desea ver? (1 = sí, ~1 = no)
graficos={
    'Referencias_y_ajustes': 1,
    'Ref_Señal_m_y_f': 1, #Sin usar
    'Ref_Señal_c_y_f': 1, #Sin usar
    'Resta_m-f': 0,
    'Resta_c-f': 0,
    'Resta_mf_y_cf':0,
    'Filtrado_calibracion': 1,
    'Filtrado_muestra': 1,
    'Recorte_a_periodos_enteros_c': 1,
    'Recorte_a_periodos_enteros_m': 1,
    'Campo_y_Mag_norm_c': 1,
    'Ciclos_HM_calibracion': 1,
    'Campo_y_Mag_norm_m': 1,
    'Ciclo_HM_m': 1 ,
    'Ciclos_HM_m_todos': 1,
    'SAR_vs_Amplitud_Campo': 0 , #Sin usar
    'SAR_vs_Amplitud_Campo**2': 0} #Sin usar

#¿Desea filtrar las señales? 
#(0 = No, 1 = Filtro Actis, 2 = Filtro Fourier)
filtrarcal = 0     # Filtro para la calibración
filtrarmuestra = 0 # Filtro para la muestra
Analisis_de_Fourier = 1 # sobre las señales, imprime espectro de señal muestra
#¿Quiere generar una imagen png con cada ciclo M vs. H obtenido? 
# escriba guarda_imagen_ciclo=1. Caso contrario, deje 0 o cualquier otro valor.
guarda_imagen_ciclo=1
concentracion = 25890 #[concentracion]= g/m^3 (1000 g/m^3 == 1 g/l) (Default = 10000 g/m^3)
mu_0 = 4*np.pi*10**-7 #[mu_0]=N/A^2
nombre_archivo_salida = 'Resultados_ESAR.dat'
textofondo = '_fondo.txt' #Texto que identifica los archivos de fondo
textocalibracion = '_cal.txt'#Texto que identifica los archivos de calibración 

#Calibracion de la bobina: cte que dimensionaliza al campo en A/m a partir de la calibracion
#realizada sobre la bobina  del RF
pendiente_HvsI = 43.18*79.77 #[pendiente_HvsI]=[Oe/A]*[1000/4*pi]=1/m
ordenada_HvsI = 2.73*79.77  #[ordenada_HvsI]=[Oe]*[1000/4*pi]=A/m
#Densidad del patron
rho_bulk_Gd2O3 = 7.41e3   #Densidad del Ox. de Gd en kg/m^3
rho_patron_Gd2O3 = 2e3   # [rho_patron_Gd2O3]= kg/m^3
#Susceptibilidad del patrón de calibración
xi_bulk_Gd2O3_masa = (1.35e-4)*4*np.pi*1e-3  #emu*m/g/A = m^3/kg
#Susceptibilidad del patrón de calibración
xi_patron_vol = xi_bulk_Gd2O3_masa*rho_patron_Gd2O3 #[xi_patron_vol]= Adimensional
#Defino listas para almacenar datos en cada iteracion
Ciclos_eje_H = []
Ciclos_eje_M = []
Ciclos_eje_H_cal = []
Ciclos_eje_M_cal = []
Ciclos_eje_H_cal_ua = []
Ciclos_eje_M_cal_ua = []
Pendiente_cal = []
Ordenada_cal = []
Pendiente_cal_filtrada = []
Ordenada_cal_filtrada = []
Frecuencia_muestra_kHz = []
Frecuencia_fondo_kHz = []
SAR = []
Campo_maximo_kAm = []
Coercitividad_kAm = []
Remanencia_kAm = []
Peor_diferencia=[]

# Imprimo configuraciones
config = {
'Selecciono todos los archivos ':bool(todos),
'Nombre archivo de muestra':nombre,
'Nombre archivo de temperaturas':nombre_T,
'Nombre archivo de salida': nombre_archivo_salida,
'Texto p/ archivo fondo' :textofondo,
'Ciclos en descongelamiento' : bool(ciclos_en_descongelamiento),
'Filtrar señal de muestra' : bool(filtrarmuestra),
'Filtrar señal de calibracion' :bool(filtrarcal),
'Analisis de Fourier' : bool(Analisis_de_Fourier), 
'Guarda imagen_ciclo':bool(guarda_imagen_ciclo),
'Concentracion (g/l)' : concentracion/1e3,
'Texto p/ archivo calibracion' : textocalibracion
}
print('Configuracion del script:')
pprint(pd.DataFrame.from_dict(data=config,orient='index'))
print('Configuracion grafica:')
pprint(pd.DataFrame.from_dict(data=graficos,orient='index'))

#Fecha para usar en graficos 
fecha_nombre = datetime.today().strftime('%Y%m%d_%H%M%S')
fecha_graf = time.strftime('%Y_%m_%d', time.localtime())
#%Cuadro de seleccion de Archivos
'''
Seleccion de carpeta con archivos via interfaz de usuario
'''
import tkinter as tk
from tkinter import filedialog
import os
import fnmatch

root = tk.Tk()
root.withdraw()

if todos==1: #Leo todos los archivos del directorio
    texto_encabezado = "Seleccionar la carpeta con las medidas a analizar:"
    directorio = filedialog.askdirectory(title=texto_encabezado)
    filenames = os.listdir(directorio) #todos

    fnames_m = []
    path_m = []
    fnames_c = []
    path_c = []
    fnames_f = []
    path_f = []

    #Seleccion de los archivos 

    if ciclos_en_descongelamiento==0:
        
        for muestra in fnmatch.filter(filenames,'*'+nombre+'.txt'):
            fnames_m.append(muestra)
            path_m.append(directorio +'/'+ muestra)
    
        for cal in fnmatch.filter(filenames,'*_cal.txt'):
            fnames_c.append(cal)
            path_c.append(directorio + '/'+ cal)

        for fondo in fnmatch.filter(filenames,'*_fondo.txt'):
            fnames_f.append(fondo)
            path_f.append(directorio + '/' + fondo)

    if ciclos_en_descongelamiento!=0:
        for muestra in fnmatch.filter(filenames,'*'+nombre_T+'*.txt'):
            fnames_m.append(muestra)
            path_m.append(directorio +'/'+ muestra)

        for cal in fnmatch.filter(filenames,'*_cal.txt'):
            l=0
            while l < len(fnames_m):
                fnames_c.append(cal)
                path_c.append(directorio + '/'+ cal)
                l+=1

        for fondo in fnmatch.filter(filenames,'*_fondo.txt'):
            m=0
            while m < len(fnames_m):
                fnames_f.append(fondo)
                path_f.append(directorio + '/' + fondo)
                m+=1  

if todos!=1: #Selecciono 1 o + archivos de muestra 
    texto_encabezado = "Seleccionar archivos con las medidas de la muestra:"
    path_m=filedialog.askopenfilenames(title=texto_encabezado,filetypes=(("Archivos .txt","*.txt"),("Archivos .dat","*.dat"),("Todos los archivos","*.*")))
    directorio = path_m[0].rsplit('/',maxsplit=1)[0]
    fa = len(path_m)
    
    fnames_m = []
    fnames_c = []
    path_c = []
    fnames_f = []
    path_f = []

    for item in path_m:    
        fnames_m.append(item.split('/')[-1])
      
    if ciclos_en_descongelamiento==0:
        for i in range(fa):
            fnames_c.append(fnames_m[i].replace('.txt',textocalibracion))     
            fnames_f.append(fnames_m[i].replace('.txt',textofondo))
            path_c.append(directorio + '/' + fnames_c[i])
            path_f.append(directorio + '/' + fnames_f[i])
    filenames = fnames_m + fnames_c+fnames_f  
    
    if ciclos_en_descongelamiento!=0:
        #busco 1 solo archivo _cal y _fondo
        filenames = os.listdir(directorio)  
        for cal in fnmatch.filter(filenames,'*_cal.txt'):
            l=0
            while l < len(fnames_m):
                fnames_c.append(cal)
                path_c.append(directorio + '/'+ cal)
                l+=1

        for fondo in fnmatch.filter(filenames,'*_fondo.txt'):
            m=0
            while m < len(fnames_m):
                fnames_f.append(fondo)
                path_f.append(directorio + '/' + fondo)
                m+=1  
# %
# Imprimo los archivos a procesar, clasificados m,c,f, y el num total
print('Directorio de trabajo: '+ directorio +'\n')
print('Archivos de muestra en el directorio: ')
for item in fnames_m:
    print(item)
print('\nArchivos de calibracion en el directorio: ')

if ciclos_en_descongelamiento==0:
    for item in fnames_c:
        print(item)
else:
    print(fnames_c[0])

print('\nArchivos de fondo en el directorio: ')
if ciclos_en_descongelamiento==0:
    for item in fnames_f:
        print(item)
    print('\nSon {} archivos.'.format(len(fnames_m)+len(fnames_c)+len(fnames_m)))    
else: 
    print(fnames_f[0])    
    print('\nSon {} archivos.'.format(len(fnames_m)+2))
print('-'*50)
# Para detectar triadas de archivos (m,c,f) incompletas
if ciclos_en_descongelamiento==0:
    if len(fnames_c)<len(fnames_m):
        raise Exception(f'Archivo de calibracion faltante\nArchivos muestra: {len(fnames_m)}\nArchivos calibracion: {len(fnames_c)} ')
    elif len(fnames_f)<len(fnames_m):
        raise Exception(f'Archivo de fondo faltante\nArchivos muestra: {len(fnames_m)}\nArchivos fondo: {len(fnames_f)}')
    elif (len(fnames_m)<len(fnames_c)) or (len(fnames_m)>len(fnames_f)):
        raise Exception(f'Archivo de muestra faltante\nArchivos muestra: {len(fnames_m)}\nArchivos calibracion: {len(fnames_c)}\nArchivos fondo: {len(fnames_f)} ')
    else:
        pass
#% Params del nombre
'''
Parámetros de la medida a partir de nombre del archivo 
de muestra: 'xxxkHz_yyydA_zzzMss_*.txt
'''
frec_nombre=[] #Frec del nombre del archivo. Luego comparo con frec ajustada
Idc = []       #Internal direct current en el generador de RF
delta_t = []   #Base temporal 
fecha_m = []   #fecha de creacion archivo, i.e., de la medida 
FMT = '%d-%m-%Y %H:%M'
for i in range(len(fnames_m)):
    frec_nombre.append(float(fnames_m[i].split('_')[0][:-3])*1000)
    Idc.append(float(fnames_m[i].split('_')[1][:-2])/10)
    delta_t.append(1e-6/float(fnames_m[i].split('_')[2][:-3]))
    fecha_m.append(datetime.fromtimestamp(os.path.getmtime(path_m[i])).strftime(FMT))

#% Procesamiento
''' 
Ejecuto medida_cruda()
En cada iteracion levanto la info de los .txt a dataframes.
''' 
for k in range(len(fnames_m)):
    '''defino DataFrames con los datos de muestra, calibracion y fondo'''
    df_m = medida_cruda(path_m[k],delta_t[k])
    df_c = medida_cruda(path_c[k],delta_t[k])
    df_f = medida_cruda(path_f[k],delta_t[k])
    '''
    Realizo el ajuste sobre la referencia y obtengo params
    '''
    offset_m , amp_m, frec_m , fase_m = ajusta_seno(df_m['t'],df_m['v_r'])
    offset_c , amp_c, frec_c , fase_c = ajusta_seno(df_c['t'],df_c['v_r'])
    offset_f , amp_f, frec_f , fase_f = ajusta_seno(df_f['t'],df_f['v_r'])

    #Genero señal simulada usando params y guardo en respectivos df
    df_m['v_r_ajustada'] = sinusoide(df_m['t'],offset_m , amp_m, frec_m , fase_m)
    df_c['v_r_ajustada'] = sinusoide(df_c['t'],offset_c , amp_c, frec_c , fase_c)
    df_f['v_r_ajustada'] = sinusoide(df_f['t'],offset_f , amp_f, frec_f , fase_f)

    df_m['residuos'] = df_m['v_r'] - df_m['v_r_ajustada']
    df_c['residuos'] = df_c['v_r'] - df_c['v_r_ajustada']
    df_f['residuos'] = df_f['v_r'] - df_f['v_r_ajustada']

    # Comparacion de señales y ajustes
    if graficos['Referencias_y_ajustes']==1:
        fig , ax = plt.subplots(3,1, figsize=(8,9),sharex='all')
        
        df_m.plot('t','v_r',label='Referencia',ax=ax[0],title='Muestra')
        df_m.plot('t','v_r_ajustada',label='Ajuste',ax =ax[0])
        df_m.plot('t','residuos', label='Residuos',ax=ax[0])
        
        df_c.plot('t','v_r',label='Referencia de Calibracion',ax=ax[1])
        df_c.plot('t','v_r_ajustada',label='Ajuste de ref. de calibracion',ax =ax[1],title='Calibración')
        df_c.plot('t','residuos', label='Residuos',ax=ax[1])

        df_f.plot('t','v_r',label='Referencia de fondo',ax=ax[2])
        df_f.plot('t','v_r_ajustada',label='Ajuste',ax =ax[2],title='Fondo')
        df_f.plot('t','residuos', label='Residuos',ax=ax[2])

        fig.suptitle('Comparacion señal de referencias y ajustes',fontsize=20)
        plt.tight_layout()
        
    # Cortafuegos: Si la diferencia entre frecuencias es muy grande => error'''
    text_error ='''
    Incompatibilidad de frecuencias en: 
            {:s}\n
        Muestra:              {:.3f} Hz
        Calibración:          {:.3f} Hz
        Fondo:                {:.3f} Hz
        En nombre de archivo: {:.3f} Hz
    '''
    incompat = np.array([abs(frec_m-frec_f)/frec_f>0.02,
                abs(frec_c-frec_f)/frec_f >0.02,
                abs(frec_c-frec_f)/frec_f >0.02,    
                abs(frec_m-frec_nombre[k])/frec_f > 0.05],dtype=bool)
    if incompat.any():
        raise Exception(text_error.format(fnames_m[k],frec_m,frec_c,frec_f,frec_nombre[k]))
    else:
        pass

    # Ejecuto la funcion resta_inter() 
    
    t_m = df_m['t'].to_numpy() #Muestra
    v_m = df_m['v'].to_numpy()
    v_r_m = df_m['v_r'].to_numpy()

    if graficos['Resta_m-f']==1:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,'muestra')
    else:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)
    
    t_c = df_c['t'].to_numpy() #Calibracion
    v_c = df_c['v'].to_numpy()
    v_r_c = df_c['v_r'].to_numpy()

    if graficos['Resta_c-f']==1:
        Resta_c , t_c_1 , v_r_c_1 , figura_c = resta_inter(t_c,v_c,v_r_c,fase_c,frec_c,offset_c,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,'calibración')
    else:
        Resta_c , t_c_1 , v_r_c_1 , figura_c = resta_inter(t_c,v_c,v_r_c,fase_c,frec_c,offset_c,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)

    # Grafico las restas 
    if graficos['Resta_mf_y_cf']==1:
        plt.figure(figsize=(10,5))
        plt.plot(t_m_1,Resta_m,'.-',lw=0.9,label='Muestra - fondo')
        plt.plot(t_c_1,Resta_c,'.-',lw=0.9,label='Calibracion - fondo')
        plt.grid()
        plt.legend(loc='best')
        plt.title('Resta de señales')
        plt.xlabel('t (s)')
        plt.ylabel('Resta (V)')
        plt.show()
    else:
        pass

    # Ejecuto filtrando_ruido()
 
    #Muestra
    if graficos['Filtrado_muestra']==1:
        t_m_2, v_r_m_2, Resta_m_2, figura_m_2 = filtrando_ruido(t_m_1,v_r_m_1,Resta_m,filtrarmuestra,'muestra')
    else:
        t_m_2, v_r_m_2, Resta_m_2, figura_m_2 = filtrando_ruido(t_m_1,v_r_m_1,Resta_m,filtrarmuestra,0)

    #Calibracion
    if graficos['Filtrado_calibracion']==1:
        t_c_2, v_r_c_2 , Resta_c_2, figura_c_2 = filtrando_ruido(t_c_1,v_r_c_1,Resta_c,filtrarcal,'calibración')
    else:
        t_c_2, v_r_c_2 , Resta_c_2, figura_c_2 = filtrando_ruido(t_c_1,v_r_c_1,Resta_c,filtrarcal,0)


    # Diferencia entre señal sin ruido y señal. Guarda el peor valor 
    interpolador_m = sc.interpolate.interp1d(t_m_1,Resta_m,'slinear')
    dif_resta_m = Resta_m_2 - interpolador_m(t_m_2)

    interpolador_c = sc.interpolate.interp1d(t_c_1,Resta_c,'slinear')
    dif_resta_c = Resta_c_2 - interpolador_c(t_c_2)

    peor_diferencia=max([np.mean(abs(dif_resta_m))/max(Resta_m),np.mean(abs(dif_resta_c))/max(Resta_c)])

    # Aplico recorte() para tener N periodos enteros 
    #Muestra
    if graficos['Recorte_a_periodos_enteros_m']==1:
        t_m_3, v_r_m_3 , Resta_m_3, N_ciclos_m, figura_m_3 = recorte(t_m_2,v_r_m_2,Resta_m_2,frec_m,'muestra')
    else:
        t_m_3, v_r_m_3 , Resta_m_3, N_ciclos_m, figura_m_3 = recorte(t_m_2,v_r_m_2,Resta_m_2,frec_m,0)
    #Calibracion
    if graficos['Recorte_a_periodos_enteros_c']==1:
        t_c_3, v_r_c_3 , Resta_c_3, N_ciclos_c, figura_c_3 = recorte(t_c_2,v_r_c_2,Resta_c_2,frec_c,'calibración')
    else:
        t_c_3, v_r_c_3 , Resta_c_3, N_ciclos_c, figura_c_3 = recorte(t_c_2,v_r_c_2,Resta_c_2,frec_c,0)

    #Ultimo ajuste sobre las señales de referencia (muestra y  cal) 
    _,_, frec_final_c,_ = ajusta_seno(t_c_3,v_r_c_3)
    _,_, frec_final_m,_ = ajusta_seno(t_m_3,v_r_m_3)

    # Ejecuto promediado_ciclos() en Calibracion
    t_f_c , fem_campo_c , R_c , delta_t_c = promediado_ciclos(t_c_3,v_r_c_3,Resta_c_3,frec_final_c,N_ciclos_c)
    
    ############################################################################################################
    ## Integro los ciclos: calcula sumas acumuladas y convierte a campo y magnetizacion
    #Cte que dimensionaliza al campo en A/m a partir de la calibracion realizada sobre la bobina del RF
    C_norm_campo=Idc[k]*pendiente_HvsI+ordenada_HvsI #[C_norm_campo]=[A]*[1/m]+[A/m]=A/m
    #Integral de la fem inducida, proporcional al Campo mas una contante
    campo_ua0_c = delta_t_c*cumulative_trapezoid(fem_campo_c,initial=0) #[campo_ua0_c]=V*s
    campo_ua_c = campo_ua0_c - np.mean(campo_ua0_c) #Resto offset
    campo_c  = (campo_ua_c/max(campo_ua_c))*C_norm_campo #[campo_c]=A/m 
    
    # Integral de la fem inducida  (c/ fondo restado), es proporcional a la magnetizacion mas una constante
    magnetizacion_ua0_c = delta_t_c*cumulative_trapezoid(R_c,initial=0)#[magnetizacion_ua0_c]=V*s
    magnetizacion_ua_c = magnetizacion_ua0_c-np.mean(magnetizacion_ua0_c)#Resto offset
    #Ajuste Lineal sobre ciclo de la calibración
    pendiente , ordenada = np.polyfit(campo_c,magnetizacion_ua_c,1) #[pendiente]=m*V*s/A  [ordenada]=V*s
    polaridad = np.sign(pendiente) # +/-1
    pendiente = pendiente*polaridad # Para que sea positiva
    magnetizacion_ua_c = magnetizacion_ua_c*polaridad  #[magnetizacion_ua_c]=V*s
    
    # Fourier
    #Analisis de Fourier sobre las señales
    if Analisis_de_Fourier == 1:
        armonicos_m,armonicos_r,amplitudes_m,amplitudes_r,fases_m,fases_r,fig_fourier, fig2_fourier, indices_m,indx_mult_m, muestra_rec_impar,cal_rec_impar,fig3_fourier,fig4_fourier,fig5_fourier,fig6_fourier = fourier_señales(t_m_3,t_c_3,Resta_m_3,Resta_c_3,v_r_m_3,v_r_c_3,delta_t[k],polaridad,filtro=0.05,frec_limite_m=30*frec_final_m,frec_limite_cal=1.5*frec_final_c,name=fnames_m[k])

        # Guardo Graficos
        fig_fourier.savefig(fnames_m[k][:-4]+'_Espectro_Fourier_{}'.format(k)+str(fecha_nombre)+'.png',dpi=200,facecolor='w')
        fig2_fourier.savefig(fnames_m[k][:-4]+'_Rec_impar_{}_'.format(k)+str(fecha_nombre)+'.png',dpi=200,facecolor='w')
        #plt.close(fig='all')   #cierro todas las figuras
        
        #Reemplazo señal recortada con la filtrada en armonicos impares en muestra y calibracion: 
            # Resta_m_3 ==> muestra_rec_impar 
            # Resta_c_3 ==> cal_rec_impar  
        t_f_m , fem_campo_m , R_m , delta_t_m = promediado_ciclos(t_m_3,v_r_m_3,muestra_rec_impar,frec_final_m,N_ciclos_m)
        t_f_c , fem_campo_c , R_c , delta_t_c = promediado_ciclos(t_c_3,v_r_c_3,cal_rec_impar,frec_final_c,N_ciclos_c)
        
        magnetizacion_ua0_c = delta_t_c*cumulative_trapezoid(R_c,initial=0)
        magnetizacion_ua_c = magnetizacion_ua0_c-np.mean(magnetizacion_ua0_c)
    else:
        #Sin analisis de Fourier, solamente acomodo la polaridad de la señal de la muestra. 
        # La señal de la calibracion no se modifica 
        t_f_m , fem_campo_m , R_m , delta_t_m = promediado_ciclos(t_m_3,v_r_m_3,Resta_m_3*polaridad,frec_final_m,N_ciclos_m)

     
    if graficos['Campo_y_Mag_norm_c']==1: # CALIBRACION: H(t) y M(t) normalizados 
        fig , ax =plt.subplots()    
        ax.plot(t_f_c,campo_c/max(campo_c),label='Campo')
        ax.plot(t_f_c,magnetizacion_ua_c/max(magnetizacion_ua_c),label='Magnetización')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Campo y magnetización normalizados del paramagneto\n'+ fnames_c[k][:-4])
  
    if graficos['Ciclos_HM_calibracion']==1: #Ciclo del paramagneto en u.a. (V*s), y Ajuste Lineal 
        fig, ax = plt.subplots()
        ax.plot(campo_c,magnetizacion_ua_c,label='Calibración')
        ax.plot(campo_c, ordenada + pendiente*campo_c,label='Ajuste lineal')
        plt.text(.6,.25,f'$M = m\cdot H + n$\nm = {pendiente:.1e} $V\cdot s\cdot m/A$\nn = {ordenada:.1e} $V \cdot s$',bbox=dict(facecolor='tab:orange',alpha=0.7),transform=ax.transAxes)
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('H $(A/m)$')
        plt.ylabel('M ($V\cdot s$)')
        plt.title('Ciclo del paramagneto\n'+ fnames_c[k][:-4])

    #Calibración para pasar la magnetización de m*V*s/A a A/m
    #Repito Ajuste Lineal sobre ciclo de la calibración, filtrado por Fourier 
    pendiente_filtrada , ordenada_filtrada = np.polyfit(campo_c,magnetizacion_ua_c,1) #[pendiente]=m*V*s/A  [ordenada]=V*s
    calibracion=xi_patron_vol/pendiente #[calibracion]=A/m*V*s
    
    # Doy unidades a la magnetizacion de calibracion, ie, al paramagneto
    magnetizacion_c = calibracion*magnetizacion_ua_c #[magnetizacion_c]=A/m
    
    #Integro los ciclos de Muestra: Calcula sumas acumuladas y convierte a campo y magnetizacion
    # Campo H(t)
    campo_ua0_m = delta_t_m*cumulative_trapezoid(fem_campo_m,initial=0) #V*s
    campo_ua_m = campo_ua0_m - np.mean(campo_ua0_m)
    campo_m = (campo_ua_m/max(campo_ua_m))*C_norm_campo #[campo_m]=A/m

    magnetizacion_ua0_m = delta_t_m*cumulative_trapezoid(R_m,initial=0)
    magnetizacion_ua_m = magnetizacion_ua0_m - np.mean(magnetizacion_ua0_m)
    magnetizacion_m=calibracion*magnetizacion_ua_m #[magnetizacion_m]=A/m
    #La polaridad se corrigió dentro de fourier_señales() o promediado_ciclos()
    
    if graficos['Campo_y_Mag_norm_m']==1: #MUESTRA: H(t) y M(t) normalizados 
        fig , ax =plt.subplots()    
        ax.plot(t_f_m,campo_ua_m/max(campo_ua_m),label='Campo')
        ax.plot(t_f_m,magnetizacion_ua_m/max(magnetizacion_ua_m),label='Magnetización')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Campo y magnetización normalizados de la muestra\n'+fnames_m[k][:-4])
    
    if graficos['Ciclo_HM_m']==1: #Ciclo de histeresis en u.a. (V*s), y Ajuste Lineal 
        fig , ax =plt.subplots()    
        ax.plot(campo_m,magnetizacion_m,label='Muestra')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('H $(A/m)$')
        plt.ylabel('M $(A/m)$')
        plt.title('Ciclo de histéresis de la muestra\n'+fnames_m[k][:-4])

    if guarda_imagen_ciclo == 1: #Guardo ciclo de histeresis M(H) individual
        fig , ax =plt.subplots()    
        ax.plot(campo_m,magnetizacion_m,label='Muestra')
        ax.plot(campo_c,magnetizacion_c,label='Paramagneto')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('Campo (A/m)')
        plt.ylabel('Magnetización (A/m)')
        plt.title('Ciclo de histéresis\n'+fnames_m[k][:-4])
        plt.savefig(fnames_m[k][:-4]+ '_ciclo_histeresis.png',dpi=300,facecolor='w')
        
    Ciclos_eje_H.append(campo_m)
    Ciclos_eje_M.append(magnetizacion_m)
    Ciclos_eje_H_cal.append(campo_c)
    Ciclos_eje_M_cal.append(magnetizacion_c)
    Ciclos_eje_M_cal_ua.append(magnetizacion_ua_c)
    Pendiente_cal.append(pendiente)
    Ordenada_cal.append(ordenada)
    Pendiente_cal_filtrada.append(pendiente_filtrada)
    Ordenada_cal_filtrada.append(ordenada_filtrada)
    #Ajuste_cal_eje_H.append()

    #% Exporto ciclos de histeresis en ASCII: Tiempo (s) || Campo (A/m) || Magnetizacion (A/m)
    
    col0 = t_f_m - t_f_m[0]
    col1 = campo_m
    col2 = magnetizacion_m
    ciclo_out = Table([col0, col1, col2])

    encabezado = ['Tiempo_(s)','Campo_(A/m)', 'Magnetizacion_(A/m)']
    formato = {'Tiempo_(s)':'%.10f' ,'Campo_(A/m)':'%f','Magnetizacion_(A/m)':'%f'} 
    ascii.write(ciclo_out,fnames_m[k][:-4] + '_Ciclo_de_histeresis.dat',
                names=encabezado,overwrite=True,delimiter='\t',formats=formato)
    
    #Calculo Coercitividad (Hc) y Remanencia (Mr) 
    m = magnetizacion_m
    h = campo_m
    Hc = []   
    Mr = [] 
    
    for z in range(0,len(m)-1):
        if ((m[z]>0 and m[z+1]<0) or (m[z]<0 and m[z+1]>0)): #M remanente
            Hc.append(abs(h[z] - m[z]*(h[z+1] - h[z])/(m[z+1]-m[z])))
            
        if((h[z]>0 and h[z+1]<0) or (h[z]<0 and h[z+1]>0)):  #H coercitivo
            Mr.append(abs(m[z] - h[z]*(m[z+1] - m[z])/(h[z+1]-h[z])))

    Hc_mean = np.mean(Hc)
    Hc_mean_kAm = Hc_mean/1000
    Hc_error = np.std(Hc)
    Mr_mean = np.mean(Mr)
    Mr_mean_kAm = Mr_mean/1000
    Mr_error = np.std(Mr)
    print(f'\nHc = {Hc_mean:.2f} (+/-) {Hc_error:.2f} (A/m)')
    print(f'Mr = {Mr_mean:.2f} (+/-) {Mr_error:.2f} (A/m)')
    #% Calculo de SAR
    #P/ la determinacion de areas armo vector aux desplazado a valores positivos
    magnetizacion_m_des = magnetizacion_m + 2*abs(min(magnetizacion_m))
    magnetizacion_c_des = magnetizacion_c + 2*abs(min(magnetizacion_c))
    
    Area_ciclo = abs(trapezoid(magnetizacion_m_des,campo_m)) 
    
    print(f'Area del ciclo de histéresis: {Area_ciclo:.2f} (A/m)^2')
    
    # Asigno el area del ciclo del paramagneto (idealmente es una recta) 
    # como la incerteza en el area del ciclo de la muestra
    Area_cal = abs(trapezoid(magnetizacion_c_des,campo_c)) #[Area_cal]=(A/m)^2
    print(f'Area del ciclo del paramagneto: {Area_cal:.2f} (A/m)^2')
     
    #Calculo de potencia disipada (SAR)
    sar = mu_0*Area_ciclo*frec_final_m/(concentracion)  #[sar]=[N/A^2]*[A^2/m^2]*[1/s]*[m^3/g]=W/g
    error_sar=abs(Area_cal)/abs(Area_ciclo) 
    #sar = ufloat(sar,error_sar) #numero con incerteza
    
    print(f'\nSAR: {sar:.2f} (W/g)')
    print(f'Concentracion: {concentracion} g/m^3')
    print(f'Fecha de la medida: {fecha_m[k]}')
    print('-'*50)
    
    #Salidas importantes: lleno las listas. Tienen tantos elmentos como archivos seleccionados    
    Frecuencia_muestra_kHz.append(frec_final_m/1000)#Frecuencia de la referencia en la medida de la muestra
    Frecuencia_fondo_kHz.append(frec_f/1000)    #Frecuencia de la referencia en la medida del fondo
    SAR.append(sar)                             #Specific Absorption Rate
    Campo_maximo_kAm.append(max(campo_m)/1000)  #Campo maximo en kA/m
    Coercitividad_kAm.append(Hc_mean/1000)      #Campo coercitivo en kA/m
    Remanencia_kAm.append(Mr_mean/1000)         #Magnetizacion remanente en kA/m
    Peor_diferencia.append(peor_diferencia*100) #Peor diferencia porcentual
    #plt.close('all')


# Plot Ciclos
if graficos['Ciclos_HM_m_todos']==1:
    fig = plt.figure(figsize=(14,8),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    axin = ax.inset_axes([0.60,0.08, 0.35,0.38])
    axin.set_title('Calibración',loc='center')
    #axin.yaxis.tick_right()
    plt.setp(axin.get_yticklabels(),visible=True)
    plt.setp(axin.get_xticklabels(),visible=True)
    axin.yaxis.tick_right()
    axin.grid()
    axin.axhline(0,0,1,lw=0.9,c='k')
    axin.axvline(0,0,1,lw=0.9,c='k')
    
    for i in range(len(fnames_m)):      
        plt.plot(Ciclos_eje_H[i],Ciclos_eje_M[i],label=f'{fnames_m[i][:-4]}\n$SAR:$ {SAR[i]:.1f} $W/g$\n{fecha_m[i]}')
        axin.plot(Ciclos_eje_H_cal[i], Ciclos_eje_M_cal_ua[i])
        axin.set_ylabel('M $(V\cdot s)$')
        axin.set_xlabel('H $(A/m)$')
        #plot(Ciclos_eje_H_cal[i], Ciclos_eje_M_cal[i],c='r')
plt.text(1.02,0.1,f'Pendiente de calibracion promedio s/filtrar:\n {ufloat(np.mean(Pendiente_cal),np.std(Pendiente_cal)):^.2ue} $V\cdot s \cdot m/A$',bbox=dict(color='tab:blue',alpha=0.8),transform=ax.transAxes)
plt.text(1.02,0.02,f'Pendiente de calibracion promedio filtrada:\n {ufloat(np.mean(Pendiente_cal_filtrada),np.std(Pendiente_cal_filtrada)):^.2ue} $V\cdot s \cdot m/A$',bbox=dict(color='tab:orange',alpha=0.8),transform=ax.transAxes)

plt.legend(loc='upper left',bbox_to_anchor=(1.01,0.5,0.4,0.5),fancybox=True)
plt.grid()
plt.xlabel('Campo (A/m)',fontsize=15)
plt.ylabel('Magnetización (A/m)',fontsize=15)
plt.title(fecha_graf,loc='left',y=0,fontsize=13)
plt.suptitle('Ciclos de histéresis',fontsize=30)
plt.savefig('Ciclos_histeresis_'+str(fecha_nombre)+'.png',dpi=300,facecolor='w')
#plt.close(fig='all')   #cierro todas las figuras 
    #%ASCII de salida

'''
Archivo de salida: utilizo las listas definidas
'''   
#Encabezado del archivo de salida 
encabezado_salida = ['Nombre del archivo analizado',
                    'Frecuencia (kHz)',
                    'Campo Maximo (kA/m)',
                    'SAR (W/g)',
                    'Coercitividad (kA/m)',
                    'Remanencia (kA/m)',
                    'Peor quita de ruido porcentual'] 
col_0 = fnames_m
col_1 = Frecuencia_muestra_kHz
col_2 = Campo_maximo_kAm
col_3 = SAR            
col_4 = Coercitividad_kAm   
col_5 = Remanencia_kAm         
col_6 = Peor_diferencia
#Armo la tabla    
salida = Table([col_0, col_1, col_2, col_3, col_4, col_5, col_6]) 
formato_salida = {'Nombre del archivo analizado':'%s',
                  'Frecuencia (kHz)':'%f',
                  'Campo Maximo (kA/m)':'%f',
                  'SAR (W/g)': '%f',
                  'Coercitividad (kA/m)': '%f',
                  'Remanencia (kA/m)': '%f', 
                  'Peor quita de ruido porcentual': '%f'} 
fecha = datetime.today().strftime('%d - %m - %Y - %H:%M:%S')

salida.meta['comments'] = [fecha, 'Concentración de la muestra: '+str(concentracion)+ ' g/m^3']

ascii.write(salida,
'Resultados_Esar_'+ str(fecha_nombre) +'.txt',
            names=encabezado_salida,
            overwrite=True,
            delimiter='\t',
            formats=formato_salida)

'''
Tiempo de procesamiento
'''

end_time = time.time()
print(f'Tiempo de ejecución del script: {(end_time-start_time):6.3f} s.')



