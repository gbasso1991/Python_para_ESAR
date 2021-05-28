#%%
# -*- coding: utf-8 -*-
"""
possessor.py 
Giuliano Andrés Basso
Aplico modificaciones a Planet_caravan_20210419.py 
basado en cactus_20210507.m
11 Mayo 2021 
"""
"""
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

Se filtra el ruido aislado de cada resta, las opciones son:
    0) No filtrar
    1) Filtro Actis: discriminando puntos donde la derivada (o menos la 
    derivada) sea alta en comparación al resto de la señal, y sus entornos. 
    En esas regiones se ajusta un polinomio con los puntos sin ruido a 
    ambos lados de la zona ruidosa (se hace lo propio en las mismas regiones 
    temporales que su señal para las respectivas referencias).
    2) Con un filtro pasabajos de Fourier.

Se recortan las señales para tener un número entero de períodos, y se
omiten tanto el primer medio período como el último. De esta manera se
evita el ruido que no pudo ser tratado adecuadamente al principio y al
final de cada medida.

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
#%% 
'''Packages'''
import time
from datetime import datetime
from numpy.core.numeric import indices 
start_time = time.time() 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sc

from scipy.signal import find_peaks 
from scipy.integrate import cumulative_trapezoid, trapezoid
#from scipy.fft import fft, ifft 
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
#%%$Funciones que usa el script
'''Funcion: fft_smooth()'''
from scipy.fft import fft, ifft
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
    Calcula seeds y ajusta sinusoide via curve_fit
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
    '''   Sin filtrar: filtrarmuestra/cal = 0
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
            plt.plot(t,v,'.-',label='Resta de señales')
            plt.plot(t_2,v_2,lw=1,label='Sin ruido')
            plt.plot(t_2,marcador,lw=1, alpha=0.8 ,label='Zona de ruido')
            #plt.plot(t,v,lw=1,label='Zona de ruido')
            plt.legend(ncol=3,loc='lower center')
            plt.grid()
            plt.xlabel('t (s)')
            plt.title('Señal de ' + str(graf),loc='left',fontsize=15)
            
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

        else: #True en la ultima iteriacion
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

#%%
'''
Input necesario de parte del usuario y definiciones preliminares
'''
#¿Qué archivos desea abrir?
#todos = 0 -> Abre el explorador para elegir los archivos 
todos=1 # Abre todos los archivos en la carpeta seleccionada
#cuyo nombre de archivo termine con el nombre de muestra:
nombre='FF'  
#¿Qué gráficos desea ver? (1 = sí, ~1 = no)
graficos={
    'Referencias_y_ajustes': 0,
    'Ref_Señal_m_y_f' : 0, #Sin usar
    'Ref_Señal_c_y_f' : 0, #Sin usar
    'Resta_m-f' : 0,
    'Resta_c-f' : 0,
    'Resta_mf_y_cf' :0,
    'Filtrado_calibracion' : 0,
    'Filtrado_muestra' : 0,
    'Recorte_a_periodos_enteros_c' : 0,
    'Recorte_a_periodos_enteros_m' : 0,
    'Campo_y_Mag_norm_c' : 0,
    'Ciclos_HM_calibracion' : 0,
    'Campo_y_Mag_norm_m' : 0,
    'Ciclo_HM_m' : 0 ,
    'Ciclos_HM_m_todos' : 1,
    'SAR_vs_Amplitud_Campo' : 0,
    'SAR_vs_Amplitud_Campo**2' : 0
    }
#¿Desea filtrar las señales? 
#(0 = No, 1 = Filtro Actis, 2 = Filtro Fourier, 3 = Filtro Fourier+Actis)
filtrarcal = 1     # Filtro para la calibración
filtrarmuestra = 1 # Filtro para la muestra
#¿Quiere generar una imagen png con cada ciclo M vs. H obtenido? 
# escriba guarda_imagen_ciclo=1. Caso contrario, deje 0 o cualquier otro valor.
guarda_imagen_ciclo=0
#Masa de nanoparticulas sobre volumen de FF en g/m^3, se utiliza para el cálculo de SAR'''
concentracion = 10000 
#Permeabilidad magnetica del vacio en N/A^2#
mu_0 = 4*np.pi*10**-7
#Nombre del archivo de salida
nombre_archivo_salida = 'resultados_ESAR.dat'
#Texto que identifica los archivos de fondo
textofondo = '_fondo.txt' #Change log
#Texto que identifica los archivos de calibración 
textocalibracion = '_cal.txt'
#Calibracion de la bobina: cte que dimensionaliza al campo en A/m a partir de la calibracion
#realizada sobre la bobina  del RF
pendiente_HvsI = 43.18*79.77 
ordenada_HvsI = 2.73*79.77  
#Susceptibilidad del patrón de calibración
rho_bulk_Gd2O3 = 7.41e3   #Densidad del Ox. de Gd en kg/m^3
rho_patron_Gd2O3 = 2e3   # kg/m^3
xi_bulk_Gd2O3_masa = (1.35e-4)*4*np.pi*1e-3  #emu*m/g/A = m^3/kg
xi_patron_vol = xi_bulk_Gd2O3_masa*rho_patron_Gd2O3
#Defino listas para almacenar datos en cada iteracion
Ciclos_eje_H = []
Ciclos_eje_M = []
Ciclos_eje_H_cal = []
Ciclos_eje_M_cal = []
Pendiente_cal = []
Ordenada_cal = []
Frecuencia_muestra_kHz = []
Frecuencia_fondo_kHz = []
SAR = []
Campo_maximo_kAm = []
Coercitividad_kAm = []
Remanencia_kAm = []
Peor_diferencia=[]

#Fecha para usar en graficos 
fecha_nombre = datetime.today().strftime('%Y%m%d_%H%M%S')
fecha_graf = time.strftime('%Y_%m_%d', time.localtime())
#%%
'''
Seleccion de carpeta con archivos via interfaz de usuario
'''
import tkinter as tk
from tkinter import filedialog
import os
import fnmatch
#%%
root = tk.Tk()
root.withdraw()
if todos==1: #Leo todos los archivos del directorio
    texto_encabezado = "Seleccionar la carpeta con las medidas a analizar:"
    directorio = filedialog.askdirectory(title=texto_encabezado)
    filenames = os.listdir(directorio) #todos

    fnames_m = []
    path_m = []
    for muestra in fnmatch.filter(filenames,'*'+nombre+'.txt'):
        fnames_m.append(muestra)
        path_m.append(directorio +'/'+ muestra)

    fnames_c = []
    path_c = []
    for cal in fnmatch.filter(filenames,'*_cal.txt'):
        fnames_c.append(cal)
        path_c.append(directorio + '/'+ cal)

    fnames_f = []
    path_f = []
    for fondo in fnmatch.filter(filenames,'*_fondo.txt'):
        fnames_f.append(fondo)
        path_f.append(directorio + '/' + fondo)

else: #Selecciono 1 o + archivos de muestra 
    texto_encabezado = "Seleccionar archivos con las medidas de la muestra:"
    path_m=filedialog.askopenfilenames(title=texto_encabezado,filetypes=(("Archivos .txt","*.txt"),("Archivos .dat","*.dat"),("Todos los archivos","*.*")))
    directorio = path_m[0].rsplit('/',maxsplit=1)[0]
    fa = len(path_m)
    
    fnames_m = []
    for item in path_m:    
        fnames_m.append(item.split('/')[-1])
    
    fnames_c = []
    path_c = []
    fnames_f = []
    path_f = []
    for i in range(fa):
        fnames_c.append(fnames_m[i].replace('.txt',textocalibracion))     
        fnames_f.append(fnames_m[i].replace('.txt',textofondo))
        path_c.append(directorio + '/' + fnames_c[i])
        path_f.append(directorio + '/' + fnames_f[i])
    filenames = fnames_m+fnames_c+fnames_f
#%%
# Imprimo los archivos a procesar, clasificados m,c,f, y el num total.
print('Directorio de trabajo: '+ directorio +'\n')
print('Archivos de muestra en el directorio: ')
for item in fnames_m:
    print(item)
print('\nArchivos de calibracion en el directorio: ')
for item in fnames_c:
    print(item)
print('\nArchivos de fondo en el directorio: ')
for item in fnames_f:
    print(item)
print(f'\nSon {len(filenames)} archivos.')
'''
Para detectar triadas de archivos (m,c,f) incompletas
Ojo con esto, para medidas en calentamiento va a cambiar
'''
if len(fnames_c)!=len(fnames_m):
    raise Exception('Falta un archivo de calibracion')
elif len(fnames_f)!=len(fnames_m):
    raise Exception('Falta un archivo de fondo')
else:
    pass
#%% 
'''
Parámetros de la medida a partir de nombre del archivo 
de muestra: 'xxxkHz_yyydA_zzzMss_nombre.txt
'''
frec_nombre=[] #Frec del nombre del archivo. Luego comparo con frec ajustada
Idc = []       #Internal direct current en el generador de RF
delta_t = []   #Base temporal 
for i in range(len(fnames_m)):
    frec_nombre.append(float(fnames_m[i].split('_')[0][:-3])*1000)
    Idc.append(float(fnames_m[i].split('_')[1][:-2])/10)
    delta_t.append(1e-6/float(fnames_m[i].split('_')[2][:-3]))

#%%
''' 
Ejecuto medida_cruda()
En cada iteracion levanto la info de los .txt a dataframes.
'''
for k in range(len(fnames_m)):
    '''defino los DataFrames con los datos de muestra, calibracion y fondo'''
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

    '''Comparacion de señales y ajustes'''
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
        
    ''' Cortafuegos: Si la diferencia entre frecuencias es muy gde => error'''
    #Agregar iterador sobre frec_nombre
    text_error ='''
    Incompatibilidad de frecuencias\n
        Muestra:           {:.3f} Hz
        Calibración:       {:.3f} Hz
        Fondo:             {:.3f} Hz
        Nombre de archivo: {:.3f} Hz
    '''
    incompat = np.array([abs(frec_m-frec_f)/frec_f>0.02,
                abs(frec_c-frec_f)/frec_f >0.02,
                abs(frec_c-frec_f)/frec_f >0.02,
                abs(frec_m-frec_nombre[0])/frec_f > 0.02],dtype=bool)
    if incompat.any():
        raise Exception(text_error.format(frec_m,frec_c,frec_f,frec_nombre[0]))
    else:
        pass

    '''Ejecuto la funcion resta_inter() '''
    #Muestra
    t_m = df_m['t'].to_numpy()
    v_m = df_m['v'].to_numpy()
    v_r_m = df_m['v_r'].to_numpy()

    if graficos['Resta_m-f']==1:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,'muestra')
    else:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)

    #Calibracion
    t_c = df_c['t'].to_numpy()
    v_c = df_c['v'].to_numpy()
    v_r_c = df_c['v_r'].to_numpy()

    if graficos['Resta_c-f']==1:
        Resta_c , t_c_1 , v_r_c_1 , figura_c = resta_inter(t_c,v_c,v_r_c,fase_c,frec_c,offset_c,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,'calibración')
    else:
        Resta_c , t_c_1 , v_r_c_1 , figura_c = resta_inter(t_c,v_c,v_r_c,fase_c,frec_c,offset_c,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)

    '''Grafico las restas'''
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

    '''
    Ejecuto filtrando_ruido()
    '''
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


    '''Diferencia entre señal sin ruido y señal. Guarda el peor valor.'''
    interpolador_m = sc.interpolate.interp1d(t_m_1,Resta_m,'slinear')
    dif_resta_m = Resta_m_2 - interpolador_m(t_m_2)

    interpolador_c = sc.interpolate.interp1d(t_c_1,Resta_c,'slinear')
    dif_resta_c = Resta_c_2 - interpolador_c(t_c_2)

    peor_diferencia=max([np.mean(abs(dif_resta_m))/max(Resta_m),np.mean(abs(dif_resta_c))/max(Resta_c)])

    
    '''Aplico recorte()'''
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

    
    '''Campo y Magnetizacion: se integran funciones de referencia y Resta.'''

    '''Calibracion:'''
    '''
    Ultimo ajuste sobre referencia
    '''
    _,_, frec_final_m,_ = ajusta_seno(t_m_3,v_r_m_3)
    _,_, frec_final_c,_ = ajusta_seno(t_c_3,v_r_c_3)

    '''
    Ejecuto promediado_ciclos()
    '''
    t_f_m , fem_campo_m , R_m , delta_t_m = promediado_ciclos(t_m_3,v_r_m_3,Resta_m_3,frec_final_m,N_ciclos_m)
    t_f_c , fem_campo_c , R_c , delta_t_c = promediado_ciclos(t_c_3,v_r_c_3,Resta_c_3,frec_final_c,N_ciclos_c)

    '''
    Integro los ciclos: calcula sumas acumuladas y convierte a campo y magnetizacion
    '''
    C_norm_campo=Idc[k]*pendiente_HvsI+ordenada_HvsI
    '''Cte que dimensionaliza al campo en A/m a partir de la calibracion
    realizada sobre la bobina del RF'''
    '''
    Integral de la fem inducida, es proporcional al campo mas una contante
    '''
    campo_ua0_c = delta_t_c*cumulative_trapezoid(fem_campo_c,initial=0)
    #Campo en volt*segundo, falta llevar a ampere/metro.
    '''
    Resto offset
    '''
    campo_ua_c = campo_ua0_c - np.mean(campo_ua0_c)
    '''
    Doy unidades al campo 
    '''
    campo_c  = (campo_ua_c/max(campo_ua_c))*C_norm_campo
    '''
    Integral de la fem inducida, fondo restado, es proporcional a
    la magnetizacion mas una constante'''
    magnetizacion_ua0_c = delta_t_c*cumulative_trapezoid(R_c,initial=0)
    #mangnetizacion en volts*segundo, falta llevar a ampere/metro.
    '''Resto offset tmb'''
    magnetizacion_ua_c = magnetizacion_ua0_c-np.mean(magnetizacion_ua0_c)
    '''
    Ajusta una recta al ciclo de la calibración
    '''
    pendiente , ordenada = np.polyfit(campo_c,magnetizacion_ua_c,1)
    polaridad = np.sign(pendiente) 

    pendiente = pendiente*polaridad
    magnetizacion_ua_c = magnetizacion_ua_c*polaridad 
   
    if graficos['Campo_y_Mag_norm_c']==1:
        fig , ax =plt.subplots()    
        ax.plot(t_f_c,campo_c/max(campo_c),label='Campo')
        ax.plot(t_f_c,magnetizacion_ua_c/max(magnetizacion_ua_c),label='Magnetización')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Campo y magnetización normalizados del paramagneto\n'+ fnames_c[k][:-4])
    if graficos['Ciclos_HM_calibracion']==1:
        fig, ax = plt.subplots()
        ax.plot(campo_c,magnetizacion_ua_c,label='Calibración')
        ax.plot(campo_c, ordenada + pendiente*campo_c,label='Ajuste lineal')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('H (Oe)')
        plt.title('Ciclo del paramagneto\n'+ fnames_c[k][:-4])

    '''Calibración para pasar la magnetización a A/m'''
    calibracion=xi_patron_vol/pendiente
    '''Doy unidades a la magnetizacion de calibracion, ie,
    al paramagneto'''
    magnetizacion_c = calibracion*magnetizacion_ua_c

    '''Integro los ciclos de Muestra: 
            Calcula sumas acumuladas y convierte a campo y magnetizacion'''
    campo_ua0_m = delta_t_m*cumulative_trapezoid(fem_campo_m,initial=0)
    campo_ua_m = campo_ua0_m - np.mean(campo_ua0_m)

    magnetizacion_ua0_m = delta_t_m*cumulative_trapezoid(R_m,initial=0)
    magnetizacion_ua_m = magnetizacion_ua0_m - np.mean(magnetizacion_ua0_m)
    magnetizacion_ua_m = polaridad*magnetizacion_ua_m
    '''
    Da unidades de A/m a la magnetizacion final utilizando la
    proporcionalidad obtenida en la calibracion. Este paso podría
    ser realizado directamente sobre un valor de calibracion
    provisto por el usuario, de calibraciones anteriores.
    '''
    magnetizacion_m=calibracion*magnetizacion_ua_m

    if graficos['Campo_y_Mag_norm_m']==1:
        fig , ax =plt.subplots()    
        ax.plot(t_f_m,campo_ua_m/max(campo_ua_m),label='Campo')
        ax.plot(t_f_m,magnetizacion_ua_m/max(magnetizacion_ua_m),label='Magnetización')
        
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Campo y magnetización normalizados de la muestra\n'+fnames_m[k][:-4])
    if graficos['Ciclo_HM_m']==1:
        fig , ax =plt.subplots()    
        ax.plot(campo_ua_m/max(campo_ua_m),magnetizacion_ua_m/max(magnetizacion_ua_m),label='Muestra')
        
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Ciclo de histéresis normalizado de la muestra\n'+fnames_m[k][:-4])

    '''Campo y magnetizacion finales '''
    campo_m = (campo_ua_m/max(campo_ua_m))*C_norm_campo
    
    '''Grafica el ciclo momentáneamente para guardar una imagen'''
    if guarda_imagen_ciclo == 1:
        fig , ax =plt.subplots()    
        ax.plot(campo_m,magnetizacion_ua_m,label='Muestra')
        ax.plot(campo_c,magnetizacion_ua_c,label='Paramagneto')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('Campo (A/m)')
        plt.ylabel('Magnetización (A/m)')
        plt.title('Ciclo de histéresis\n'+fnames_m[k][:-4])
        plt.savefig(fnames_m[k][:-4]+ '_ciclo_histeresis.png',dpi=300)

    Ciclos_eje_H.append(campo_m)
    Ciclos_eje_M.append(magnetizacion_m)

    Ciclos_eje_H_cal.append(campo_c)
    Ciclos_eje_M_cal.append(magnetizacion_c)
    Pendiente_cal.append(pendiente)
    Ordenada_cal.append(ordenada)
    #Ajuste_cal_eje_H.append()

    #%%
    '''Exporto ciclos de histeresis en ascii: Tiempo (s) || Campo (A/m) || Magnetizacion (A/m)'''
 
    col0 = t_f_m - t_f_m[0]
    col1 = campo_m
    col2 = magnetizacion_m
    ciclo_out = Table([col0, col1, col2])

    encabezado = ['Tiempo_(s)','Campo_(A/m)', 'Magnetizacion_(A/m)']
    formato = {'Tiempo_(s)':'%.10f' ,'Campo_(A/m)':'%f','Magnetizacion_(A/m)':'%f'} 
    ascii.write(ciclo_out,fnames_m[k][:-4] + '_Ciclo_de_histeresis.dat',
                names=encabezado,overwrite=True,delimiter='\t',formats=formato)
    
    '''Calculo Coercitividad (Hc) y Remanencia (Mr) '''
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
    
    print(f'Hc = {Hc_mean:.2f} (+/-) {Hc_error:.2f} (A/m)')
    print(f'Mr = {Mr_mean:.2f} (+/-) {Mr_error:.2f} (A/m)')

    
    '''Calculo de SAR''' 
    '''
    Determinacion de areas: armo vector aux desplazado a valores positivos
    '''
    magnetizacion_m_des = magnetizacion_m + 2*abs(min(magnetizacion_m))
    magnetizacion_c_des = magnetizacion_c + 2*abs(min(magnetizacion_c))
    '''
    Area del lazo
    '''
    Area_ciclo = abs(trapezoid(magnetizacion_m_des,campo_m)) 
    print('Archivo: %s' %fnames_m[k])
    print('Area del ciclo: %f' %Area_ciclo)
    '''
    Area de la calibracion: es la incerteza en el area del lazo
    '''
    Area_cal = abs(trapezoid(magnetizacion_c_des,campo_c))
    '''
    Calculo de potencia disipada SAR
    '''
    sar = mu_0*Area_ciclo*frec_final_m/(concentracion)  
    #para pasar de concentracion en g/l a g/m^3 debo considerar factor 1000 
    error_sar=100*abs(Area_cal)/abs(Area_ciclo) #porcentual
    print('''SAR: {:.2f} (W/g), incerteza del {:.2f}%
    Concentración: {} g/m^3
    '''.format(sar,error_sar,concentracion))

    '''Salidas importantes: lleno las listas. 
    Tienen tantos elmentos como archivos seleccionados'''    
    
    ''' Frecuencia de la referencia en la medida de la muestra'''
    Frecuencia_muestra_kHz.append(frec_final_m/1000)
    ''' Frecuencia de la referencia en la medida del fondo'''
    Frecuencia_fondo_kHz.append(frec_f/1000)
    ''' Specific Absorption Rate'''
    SAR.append(sar)
    ''' Campo maximo en kA/m'''
    Campo_maximo_kAm.append(C_norm_campo/1000) 
    ''' Campo coercitivo en kA/m'''
    Coercitividad_kAm.append(Hc_mean/1000)
    ''' Magnetizacion remanente en kA/m'''
    Remanencia_kAm.append(Mr_mean/1000)
    '''Peor diferencia porcentual'''
    Peor_diferencia.append(peor_diferencia*100)

#%%
if graficos['Ciclos_HM_m_todos']==1:
    fig = plt.figure(figsize=(10,8),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)
    axin = ax.inset_axes([0.63,0.05, 0.35,0.35])
    axin.set_title('Calibración ',loc='center')
    #axin.yaxis.tick_right()
    plt.setp(axin.get_yticklabels(),visible=False)
    plt.setp(axin.get_xticklabels(),visible=True)
    axin.grid()
    axin.axhline(0,0,1,lw=0.9,c='k')
    axin.axvline(0,0,1,lw=0.9,c='k')
    
    for i in range(len(fnames_m)):      
        plt.plot(Ciclos_eje_H[i],Ciclos_eje_M[i],label=fnames_m[i][:-4])
        axin.plot(Ciclos_eje_H_cal[i], Ciclos_eje_M_cal[i])
    
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('Campo (A/m)',fontsize=15)
    plt.ylabel('Magnetización (A/m)',fontsize=15)
    plt.title(fecha_graf,fontsize=15)
    plt.suptitle('Ciclos de histéresis',fontsize=20)
    plt.savefig('Ciclos_histeresis_'+str(fecha_nombre)+'.png',dpi=300)
#%%
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

ascii.write(salida,'Resultados_Esar_'+ str(fecha_nombre) +'.txt',names=encabezado_salida,overwrite=True,
            delimiter='\t',formats=formato_salida)

'''
Tiempo de procesamiento
'''
end_time = time.time()
print(f'Tiempo de ejecución del script: {(end_time-start_time):6.3f} s.')


