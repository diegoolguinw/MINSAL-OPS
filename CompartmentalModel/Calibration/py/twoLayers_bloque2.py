# Importanción de modulos
from datetime import datetime
import os
import hashlib
import pickle
import re
import numpy as np
import scipy as sp
import pandas as pd
import pystan
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Uncomment if you want to use multiprocessing
#import multiprocessing
#multiprocessing.set_start_method("fork", force=True)

jobid = sys.argv[1]
stanmod = sys.argv[2]
gamma_aN = float(sys.argv[3])
gamma_sN = float(sys.argv[4])
deltaN = float(sys.argv[5])
mes = sys.argv[6]
inputfile = sys.argv[7]

#inputfl = pd.read_csv(inputfile).transpose()
with open(inputfile) as f:
    inputfl = dict(filter(None, csv.reader(f)))

S0N_anterior = float(inputfl['S0Nf'])
S0V_anterior = float(inputfl['S0Vf'])
I_a0N_anterior = float(inputfl['I_a0Nf'])
I_s0N_anterior = float(inputfl['I_s0Nf'])
R_a0N_anterior = float(inputfl['R_a0Nf'])
R_s0N_anterior = float(inputfl['R_s0Nf'])
I_a0V_anterior = float(inputfl['I_a0Vf'])
I_s0V_anterior = float(inputfl['I_s0Vf'])
R_a0V_anterior = float(inputfl['R_a0Vf'])


print(f"S0N_anterior: {S0N_anterior}\n")
print(f"S0V_anterior: {S0V_anterior}\n")
print(f"I_a0N_anterior: {I_a0N_anterior}\n")
print(f"I_a0V_anterior: {I_a0V_anterior}\n")
print(f"I_s0N_anterior: {I_s0N_anterior}\n")
print(f"I_s0V_anterior: {I_s0V_anterior}\n")
print(f"R_a0N_anterior: {R_a0N_anterior}\n")
print(f"R_a0V_anterior: {R_a0V_anterior}\n")
print(f"R_s0N_anterior: {R_s0N_anterior}\n")


#pathOutput = "output/twoLayers/"
pathOutput = os.path.join("output/twoLayers/", jobid)
os.mkdir(pathOutput)
pathStan = "stanModels/twoLayers/"

sns.set()
os.chdir('./')


# compile the model (uncomment if needed)
#sm = pystan.StanModel(file=pathStan+"twoLayers.stan")
sm = pystan.StanModel(file=pathStan+stanmod)

# save the model to prevent compilation the next time
# comment if already compiled
#with open(pathStan+'twoLayers.pkl', 'wb') as f:
#    pickle.dump(sm, f)

# load the model if already compiled and regitered
#with open(pathStan+'twoLayers.pkl', 'rb') as f:
#    sm = pickle.load(f)
    
 # Datos para el modelo
# fechas para entrenar
#t0 = '2021-03-01'
#tf = '2021-03-31'
# fechas para validar
#tval = '2021-04-01'
# fecha para predecir
#tpred = '2021-04-07'

#ABRIL
if mes == "abril":
  t0 = '2021-04-01'
  tf = '2021-08-15'
  tval = '2021-08-15'
  tpred = '2021-08-15'

#MAYO
elif mes == "mayo":
  t0 = '2021-05-01'
  tf = '2021-05-31'
  tval = '2021-06-01'
  tpred = '2021-06-07'

#JUNIO
elif mes == "junio":
  t0 = '2021-06-01'
  tf = '2021-06-30'
  tval = '2021-07-01'
  tpred = '2021-07-07'

#JULIO
elif mes == "julio":
  t0 = '2021-07-01'
  tf = '2021-07-31'
  tval = '2021-08-01'
  tpred = '2021-08-07'

#AGOSTO
elif mes == "agosto":
  t0 = '2021-08-01'
  tf = '2021-08-30'
  tval = '2021-09-01'
  tpred = '2021-09-07'


# cargar el archivo de datos
data = pd.read_csv('input/chileconvacunas3.csv', index_col=0)
data.index = pd.to_datetime(data.index, dayfirst=False)
data_tr = data[t0:tf]
data_val = data[tf:tval]
data_pred = data[t0:tpred]

# tiempo de entreno
T = np.arange(0, len(data_tr))

# tiempo de validación
T_val = np.arange(len(data_tr), len(data_tr)+len(data_val))

# tiempo de predicción
T_pred = np.arange(0, len(data_pred))

# population in Chile (projection for 2021 with the census of 2017)
N0 = 19458310
# %pob contagiada y aislada
mu = 0.5
# rate of lockdown
ut = 0.5
# rate of ...
q_aN = 1/480
q_sN = 1/480
pN = 0.3
q_aV = 1/480
q_sV = 1/480
pV = 0.7

# Vector con los datos
y = np.zeros((len(T),14))
y[:,5] = data_tr['Fallecidos novac']
y[:,6] = data_tr['Casos totales']
y[:,12] = data_tr['Fallecidos vac']

ye=data_pred['Efectividad']
yVd0=data_pred['Vacunados']
yVd=np.zeros_like(yVd0)

for i in range(len(yVd)-1):
    yVd[i]=yVd0[i+1]-yVd0[i]

yVd[-1]=yVd[-2]
#vacunados diarios promedio en el periodo con segundas dosis
#v=113848
#Entrega a stan
stan_data = {
	    'N': len(T),
	    'T': T,
	    'N_pred': len(T_pred),
	    'T_pred': T_pred,
	    'N0': N0,
	    'mu': mu,
	    'ut': ut,
	    'q_aN': q_aN,
	    'q_sN': q_sN,
	    'pN': pN,
	    'q_aV': q_aV,
	    'q_sV': q_sV,
	    'pV': pV,
	    'yD1': y[:,5],
	    'yIs': y[:,6],
    	    'yD2': y[:,12],
	    'yV': yVd,
	    'ye':ye,
	    'gamma_aN': gamma_aN,
            'gamma_sN': gamma_sN,
	    'deltaN': deltaN,
            'S0N_anterior':S0N_anterior,
            'S0V_anterior':S0V_anterior,
            'I_a0N_anterior':I_a0N_anterior,
            'I_s0N_anterior':I_s0N_anterior,
            'R_a0N_anterior':R_a0N_anterior,
            'R_s0N_anterior':R_s0N_anterior,
            'I_a0V_anterior':I_a0V_anterior,
            'I_s0V_anterior':I_s0V_anterior,
            'R_a0V_anterior':R_a0V_anterior,

}

# Condiciones para el modelo y correrlo
#warmup el default que se utilizó antes era iter/2
#chains debería ser igual al número de cores

#fit = sm.sampling(data=stan_data, iter=1000, warmup=500, chains=10, check_hmc_diagnostics=False, control={'max_treedepth': 14})
fit = sm.sampling(data=stan_data, iter=200, warmup=100, chains=10, check_hmc_diagnostics=False, control={'max_treedepth': 14})


pystan.check_hmc_diagnostics(fit, verbose=True)
summary = fit.stansummary()
z_last = re.findall("z\[%d,.*" % (len(T),), summary)
summary = re.sub("(y_pred|z|theta)\[.*\n?", "", summary)
print(summary)

print()
for z in z_last:
    print(z)
print()

# Obtención de soluciones
params = ['gamma_aV', 'gamma_sV','deltaV', 'beta_aN','beta_sN','beta_aV','beta_sV','S0N','S0V','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','I_a0V','I_s0V','R_a0V','R_s0V','D0V','subr','S0Nf','I_a0Nf','I_s0Nf','R_a0Nf','R_s0Nf','S0Vf','I_a0Vf','I_s0Vf','R_a0Vf','R_s0Vf','D0Nf','D0Vf']

means = np.zeros([1,len(params)]) # +1 para agregar S0f

for j in range(len(params)-12):
    param = fit[params[j]]
    means[0,j] = np.mean(param)

y_pred2 = fit.extract(['y_pred'])['y_pred']
pred = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
S0Nf = pred[1,-1,0] # S0 vacunados FINAL PARA EL SIGUIENTE MES
I_a0Nf = pred[1,-1,1]
I_s0Nf = pred[1,-1,2]
R_a0Nf = pred[1,-1,3]
R_s0Nf = pred[1,-1,4]
S0Vf = pred[1,-1,7] # S0 vacunados FINAL PARA EL SIGUIENTE MES
I_a0Vf = pred[1,-1,8]
I_s0Vf = pred[1,-1,9]
R_a0Vf = pred[1,-1,10]
R_s0Vf = pred[1,-1,11]
D0Nf = pred[1,-1,5] # S0 vacunados FINAL PARA EL SIGUIENTE MES
D0Vf = pred[1,-1,12] # S0 vacunados FINAL PARA EL SIGUIENTE MES

means[0,-12] = S0Nf
means[0,-11] = I_a0Nf
means[0,-10] = I_s0Nf
means[0,-9] = R_a0Nf
means[0,-8] = R_s0Nf
means[0,-7] = S0Vf
means[0,-6] = I_a0Vf
means[0,-5] = I_s0Vf
means[0,-4] = R_a0Vf
means[0,-3] = R_s0Vf
means[0,-2] = D0Nf
means[0,-1] = D0Vf

ox=pd.DataFrame(means,columns=params)
ox2 = ox.T
ox2.to_csv(pathOutput+'/tasas_means.csv')


#Ploteo
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6','C7','C8','C9','C0','C1','C2','C3','C4','C5']
names = ['SN(t)', 'I_aN(t)', 'I_sN(t)', 'R_aN(t)', 'R_sN(t)', 'DN(t)','I_ac_novac(t)','SV(t)', 'I_aV(t)', 'I_sV(t)', 'R_aV(t)', 'R_sV(t)', 'DV(t)', 'I_ac_vac']
#ploteo solución completa
def plot_odepiso1(t_obs, obs, t_pred, param,t_val,val):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(20,20), squeeze=False)
    fig.suptitle('MCMC fit primer piso {t0} – {tf}')
    aj=['Fallecidos','Casos totales']
    con=0
    for k in range(6):
        ax = axs[int(k/3),k%3]
        #if k in [5,6]:
         #   ax.plot(t_obs, obs[:,k], color=colors[k], marker='o', linestyle='none', ms=3)
          #  ax.plot(t_val,val[aj[con]],color='r',linestyle='none',marker='o',ms=3)
           # con=con+1
        ax.plot(t_pred, pred[0,:,k], color=colors[k], ls='--')
        ax.plot(t_pred, pred[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
        ax.plot(t_pred, pred[2,:,k], color=colors[k], ls='--')
        ax.legend()

def plot_odepiso2(t_obs, obs, t_pred, param,t_val,val):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(20,20), squeeze=False)
    fig.suptitle('MCMC fit segundo piso {t0} – {tf}')
    aj=['Fallecidos','Casos totales']
    con=0
    for k in range(6):
        ax = axs[int(k/3),k%3]
        #if k in [5,6]:
         #   ax.plot(t_obs, obs[:,k], color=colors[k], marker='o', linestyle='none', ms=3)
          #  ax.plot(t_val,val[aj[con]],color='r',linestyle='none',marker='o',ms=3)
           # con=con+1
        ax.plot(t_pred, pred[0,:,k+7], color=colors[k], ls='--')
        ax.plot(t_pred, pred[1,:,k+7], color=colors[k], ls='-', label='$%s$'%names[k+7])
        ax.plot(t_pred, pred[2,:,k+7], color=colors[k], ls='--')
        ax.legend()


print('Plotting prediction')
y_pred2 = fit.extract(['y_pred'])['y_pred']
T_pred1=T_pred

plot_odepiso1(T, y, T_pred1, y_pred2,T_val,data_val)
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/pred_1.png', bbox_inches='tight')
plt.show()
plot_odepiso2(T, y, T_pred1, y_pred2,T_val,data_val)
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/pred_2.png', bbox_inches='tight')
plt.show()

#
dataplot=pd.date_range(start=t0,end=tf)
dataplot2=pd.date_range(start=tf,end=tval)
dataplot3=pd.date_range(start=t0,end=tpred)

#D
pred2 = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
k=5
plt.figure()
plt.title('Fallecidos no vacunados')
plt.plot(dataplot, y[:,k], color='k', marker='o', linestyle='none', ms=3)
plt.plot(dataplot2,data_val['Fallecidos novac'],color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot3, pred2[0,:,k], color=colors[k], ls='--')
plt.plot(dataplot3, pred2[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
plt.plot(dataplot3, pred2[2,:,k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/muertesnovac.png')
plt.show()

#Acumulados
pred2 = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
k=6
plt.figure()
plt.title('Casos totales')
plt.plot(dataplot, y[:,k], color='k', marker='o', linestyle='none', ms=3)
#plt.plot(dataplot2,data_val['Casos totales']/sub,color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot2,data_val['Casos totales'],color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot3, pred2[0,:,k], color=colors[k], ls='--')
plt.plot(dataplot3, pred2[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
plt.plot(dataplot3, pred2[2,:,k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/Casos totalesnovac.png')
plt.show()

#D
pred2 = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
k=12
plt.figure()
plt.title('Fallecidos vacunados')
plt.plot(dataplot, y[:,k], color='k', marker='o', linestyle='none', ms=3)
plt.plot(dataplot2,data_val['Fallecidos vac'],color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot3, pred2[0,:,k], color=colors[k], ls='--')
plt.plot(dataplot3, pred2[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
plt.plot(dataplot3, pred2[2,:,k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/muertesvac.png')
plt.show()

