# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

from uncertainties import ufloat, unumpy

C = ufloat(52.5,0.02)
PM=(15.9994*4)+(55.845*3)
Concentracion = C*PM/(3*55.845)
print(f'Concentracion: {Concentracion} mg/ml')

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
df2 = pd.DataFrame(np.random.rand(10, 4), columns=list("WXYZ"))
df3 = pd.DataFrame(np.random.rand(10, 4), columns=list("EFGH"))
df4 = pd.DataFrame(np.random.rand(10, 4), columns=list("IJKL"))


fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2)
fig.subplots_adjust(wspace=0.01)

sns.heatmap(df1, cmap="plasma", ax=ax1, cbar=False)
sns.heatmap(df2, cmap="hot", ax=ax2, cbar=False)
sns.heatmap(df3, cmap="plasma", ax=ax3, cbar=False)
sns.heatmap(df4, cmap="hot", ax=ax4, cbar=False)


ax2.yaxis.tick_right()

fig.subplots_adjust(wspace=0.001)
plt.show()


#%%%



    return armonicos, armonicos_r, amplitudes, amplitudes_r, fases , fases_r , fig, fig2, indices, indx_impar, rec_impares,rec_impares_c,fig3,fig4,fig5,fig6,  t,y, rec_impares, rec_pares ,armonicos[0],t_c,y_c,rec_impares_c,rec_pares_c,armonicos_c,f_impar_c,amp_impar_c,armonicos_r,f_impar ,amp_impar,f

,   t_aux,y_aux,rec_impares_aux,rec_pares_aux,armonicos_0_aux,t_c_aux,y_c_aux,rec_impares_c_aux,rec_pares_c_aux,armonicos_c_aux,f_impar_c_aux,amp_impar_c_aux,armonicos_r_aux,f_impar_aux,amp_impar_aux,f_aux