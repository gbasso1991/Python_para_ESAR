# %%
'''
Lector archivos VSM. El equipo registra en Gauss y emu
Giuliano Basso
1º Jul 22
'''

import numpy as np
import matplotlib.pyplot as plt

top_path = 'ResATop.txt'
masa_top=0.0777 #g
center_path = 'ResACenter.txt'
masa_center=0.0799 #g
bottom_path = 'ResABottom.txt'
masa_bottom= 0.0927 #g

campo_top, mom_magnet_top =  np.loadtxt(top_path,dtype='float',skiprows=12,usecols=(0,1),unpack=True)
campo_top=campo_top*(1000/(4*np.pi))#[A/m]
mass_magnet_top= (mom_magnet_top/masa_top)#[emu/g]==[Am^2/kg]

campo_center, mom_magnet_center =  np.loadtxt(center_path,dtype='float',skiprows=12,usecols=(0,1),unpack=True)
campo_center=campo_center*1000/(4*np.pi)#[A/m]
mass_magnet_center= (mom_magnet_center/masa_center)#[emu/g]==[Am^2/kg]

campo_bottom, mom_magnet_bottom =  np.loadtxt(bottom_path,dtype='float',skiprows=12,usecols=(0,1),unpack=True)
campo_bottom=campo_bottom*1000/(4*np.pi)#[A/m]
mass_magnet_bottom= (mom_magnet_bottom/masa_bottom)#[emu/g]==[Am^2/kg]


plt.plot(campo_top,mass_magnet_top,'.-',label='Top')
plt.axhline(mass_magnet_top[0],0,1, label=f'{mass_magnet_top[0]:.3f} Am$^2$/kg')

plt.plot(campo_center,mass_magnet_center,'.-',label='Center')
plt.axhline(max(mass_magnet_center),0,1,color='tab:orange', label=f'{max(mass_magnet_center):.3f} Am$^2$/kg')

plt.plot(campo_bottom,mass_magnet_bottom,'.-',label='Bottom')
plt.axhline(max(mass_magnet_bottom),0,1,color='tab:green', label=f'{max(mass_magnet_bottom):.3f} Am$^2$/kg')


plt.grid()
plt.xlabel('Campo (A/m)')
plt.ylabel('Magnetizacion Masica (Am$^2$/kg)')
plt.title('VSM en Ferroresina A')
plt.xlim(1e6,1.6e6)
plt.ylim(0.83,0.855)
plt.legend( bbox_to_anchor=(1.01, 1))


plt.show()