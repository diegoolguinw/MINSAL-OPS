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

# 1. job id (desde el sistema)
# 2. modelo stan a utilizar
# 3. mes
# 4. gamma_aN
# 5. gamma_sN
# 6. deltaN
# 7. input file
jobid = sys.argv[1]
stanmod = sys.argv[2]
mes = sys.argv[3]
#gamma_aN = float(sys.argv[4])
#gamma_sN = float(sys.argv[5])
#deltaN = float(sys.argv[6])
inputfile = sys.argv[7]


with open(inputfile) as f:
    inputfl = dict(filter(None, csv.reader(f)))

gamma_aN = float(inputfl['gamma_aN'])
gamma_sN = float(inputfl['gamma_sN'])
deltaN = float(inputfl['deltaN'])
gamma_aV = float(inputfl['gamma_aV'])
gamma_sV = float(inputfl['gamma_sV'])
deltaV = float(inputfl['deltaV'])
I_a0N_anterior = float(inputfl['I_aNf'])
I_s0N_anterior = float(inputfl['I_sNf'])
R_a0N_anterior = float(inputfl['R_aNf'])
R_s0N_anterior = float(inputfl['R_sNf'])
I_a0V_anterior = float(inputfl['I_aVf'])
I_s0V_anterior = float(inputfl['I_sVf'])
R_a0V_anterior = float(inputfl['R_aVf'])
D0N_anterior = float(inputfl['DNf'])
R_s0V_anterior = float(inputfl['R_sVf'])
D0V_anterior = float(inputfl['DVf'])

S0N_anterior = float(inputfl['SNf']) # S0 final del mes anterior (15/08) no vacunados
S0V_anterior = float(inputfl['SVf']) # S0 final del mes anterior (15/08) vacunados #float(inputfl['S0Vf'])

#S0Vr_anterior = 260246 # vacunos con refuerzo al 15/08
S0Vr_anterior = float(inputfl['SVrf'])
I_a0Vr_anterior = float(inputfl['I_aVrf'])
I_s0Vr_anterior = float(inputfl['I_sVrf'])
R_a0Vr_anterior = float(inputfl['R_aVrf'])
R_s0Vr_anterior = float(inputfl['R_sVrf'])
D0Vr_anterior = float(inputfl['DVrf'])

#I_a0Vr_anterior = 0
#I_s0Vr_anterior = 0
#R_a0Vr_anterior = 0
#R_s0Vr_anterior = 0
#D0Vr_anterior = 0

pathOutput = os.path.join("output/threeLayers/", jobid)
os.mkdir(pathOutput)
pathStan = "stanModels/threeLayers/"

sns.set()
os.chdir('./')

# compile the model (uncomment if needed)
#sm = pystan.StanModel(file=pathStan+"twoLayers.stan")
sm = pystan.StanModel(file=pathStan+stanmod)

# save the model to prevent compilation the next time
# comment if already compiled
#

#with open(pathStan+'twoLayers.pkl', 'wb') as f:
#    pickle.dump(sm, f)

# load the model if already compiled and regitered
#with open(pathStan+'twoLayers.pkl', 'rb') as f:
#    sm = pickle.load(f)

if mes == "octubre-dic":
  t0 = '2021-10-01'
  tf = '2021-12-31'
  tval = '2021-12-31'
  tpred = '2021-12-31'

if mes == "15oct-dic":
  t0 = '2021-10-16'
  tf = '2021-12-31'
  tval = '2021-12-31'
  tpred = '2021-12-31'

# agregar
# desde 16 agosto hasta 31 oct y del 1 nov al 31 de diciembre


# cargar el archivo de datos
data = pd.read_csv('input/chileconvacunas.csv', index_col=0)
data.index = pd.to_datetime(data.index, dayfirst=True)
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
#N0 = 19458310
N0 = 20000000
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
# rate of ...
q_aN = 1/480
q_sN = 1/480
pN = 0.3
q_aV = 1/480
q_sV = 1/480
pV = 0.7
q_aVr = 1/480
q_sVr = 1/480
pVr = 0.7

# Vector con los datos
y = np.zeros((len(T),20))
y[:,6] = data_tr['Cases']
y[:,13] = data_tr['Deaths']

ye=data_pred['Efectividad']
ye2=data_pred['Efe refuerzo']
yVd0=data_pred['Vacunados']
yVrd0=data_pred['refuerzo']
yVd=np.zeros_like(yVd0)
yVrd=np.zeros_like(yVrd0)

for i in range(len(yVd)-1):
    yVd[i]=yVd0[i+1]-yVd0[i]
    yVrd[i]=yVrd0[i+1]-yVrd0[i]
    
yVd[-1]=yVd[-2]
yVrd[-1]=yVrd[-2]

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
	    'q_aVr': q_aVr,
	    'q_sVr': q_sVr,
	    'pVr': pVr,
	    'yIs': y[:,6],
	    'yD': y[:,13],
	    'yV': yVd,
	    'ye':ye,
	    'yVr': yVrd,
	    'yer':ye2,
	    'gamma_aN': gamma_aN,
            'gamma_sN': gamma_sN,
	    'deltaN': deltaN,
	    'gamma_aV': gamma_aV,
            'gamma_sV': gamma_sV,
	    'deltaV': deltaV,
            'S0N_anterior': S0N_anterior,
            'D0N_anterior': D0N_anterior,
            'I_a0N_anterior': I_a0N_anterior,
            'I_s0N_anterior': I_s0N_anterior,
            'R_a0N_anterior': R_a0N_anterior,
            'R_s0N_anterior': R_s0N_anterior,
            'S0V_anterior': S0V_anterior, 
            'I_a0V_anterior': I_a0V_anterior,
            'I_s0V_anterior': I_s0V_anterior,
            'R_a0V_anterior': R_a0V_anterior,
            'R_s0V_anterior': R_s0V_anterior,
            'D0V_anterior': D0V_anterior,
            'S0Vr_anterior': S0Vr_anterior,
	    'I_a0Vr_anterior': I_a0Vr_anterior,
            'I_s0Vr_anterior': I_s0Vr_anterior,
            'R_a0Vr_anterior': R_a0Vr_anterior,
            'R_s0Vr_anterior': R_s0Vr_anterior,
	    'D0Vr_anterior': D0Vr_anterior,
}

fit = sm.sampling(data=stan_data, iter=100, warmup=50, chains=10, check_hmc_diagnostics=False, control={'max_treedepth': 11})
#fit = sm.sampling(data=stan_data, iter=1000, warmup=500, chains=10, check_hmc_diagnostics=False, control={'max_treedepth': 11})

pystan.check_hmc_diagnostics(fit, verbose=True)
summary = fit.stansummary()
z_last = re.findall("z\[%d,.*" % (len(T),), summary)
summary = re.sub("(y_pred|z|theta)\[.*\n?", "", summary)
print(summary)

print()
for z in z_last:
    print(z)
print()

#Ploteo
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6','C7','C8','C9','C0','C1','C2','C3','C4','C5','C0','C1','C2','C3']
names = ['SN(t)', 'I_aN(t)', 'I_sN(t)', 'R_aN(t)', 'R_sN(t)', 'DN(t)','Ac(t)','SV(t)', 'I_aV(t)', 'I_sV(t)', 'R_aV(t)', 'R_sV(t)', 'DV(t)', 'D(t)','SVr(t)', 'I_aVr(t)', 'I_sVr(t)', 'R_aVr(t)', 'R_sVr(t)', 'DVr(t)']

#ploteo solución completa
def plot_odepiso1(t_obs, obs, t_pred, param,t_val,val):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(20,20), squeeze=False)
    fig.suptitle('MCMC fit primer piso ({t0} – {tpred})')
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
    fig.suptitle('MCMC fit segundo piso ({t0} – {tpred})')
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

def plot_odepiso3(t_obs, obs, t_pred, param,t_val,val):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(20,20), squeeze=False)
    fig.suptitle('MCMC fit tercer piso ({t0} – {tpred})')
    aj=['Fallecidos','Casos totales']
    con=0
    for k in range(6):
        ax = axs[int(k/3),k%3]
        #if k in [5,6]:
         #   ax.plot(t_obs, obs[:,k], color=colors[k], marker='o', linestyle='none', ms=3)
         #  ax.plot(t_val,val[aj[con]],color='r',linestyle='none',marker='o',ms=3)
           # con=con+1
        ax.plot(t_pred, pred[0,:,k+14], color=colors[k], ls='--')
        ax.plot(t_pred, pred[1,:,k+14], color=colors[k], ls='-', label='$%s$'%names[k+14])
        ax.plot(t_pred, pred[2,:,k+14], color=colors[k], ls='--')
        ax.legend()


# Obtención de soluciones
#params i= ['gamma_aN', 'gamma_sN','gamma_aV', 'gamma_sV', 'deltaN','deltaV', 'beta_aN','beta_sN','beta_aV','beta_sV','S0N','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','S0V','I_a0V','I_s0V','R_a0V','R_s0V','D0V','D0','RAc0']
#cambiar acá si no se fija el subreporte
#params = ['gamma_aN', 'gamma_sN','gamma_aV', 'gamma_sV', 'deltaN','deltaV', 'beta_aN','beta_sN','beta_aV','beta_sV','S0N','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','S0V','I_a0V','I_s0V','R_a0V','R_s0V','D0V','D0','RAc0','subr']
#params = ['gamma_aV', 'gamma_sV','deltaV', 'beta_aN','beta_sN','beta_aV','beta_sV','S0N','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','S0V','I_a0V','I_s0V','R_a0V','R_s0V','D0V','D0','RAc0','subr']
#params = ['gamma_aVr', 'gamma_sVr','deltaVr','beta_aN','beta_sN','beta_aV','beta_sV','beta_aVr','beta_sVr','S0N','S0V','S0Vr','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','I_a0V','I_s0V','R_a0V','R_s0V','D0V','I_a0Vr','I_s0Vr','R_a0Vr','R_s0Vr','D0Vr','D0','subr','gamma_aN', 'gamma_sN','deltaN','gamma_aV', 'gamma_sV','deltaV','S0Nf','I_a0Nf','I_s0Nf','R_a0Nf','R_s0Nf','S0Vf','I_a0Vf','I_s0Vf','R_a0Vf','R_s0Vf','S0Vrf','I_a0Vrf','I_s0Vrf','R_a0Vrf','R_s0Vrf','D0Nf','D0Vf','D0Vrf','D0f']

params = ['gamma_aVr', 'gamma_sVr','deltaVr','beta_aN','beta_sN','beta_aV','beta_sV','beta_aVr','beta_sVr','subr','S0N','I_a0N','I_s0N','R_a0N','R_s0N','D0N','Ac0','S0V','I_a0V','I_s0V','R_a0V','R_s0V','D0V','D0','S0Vr','I_a0Vr','I_s0Vr','R_a0Vr','R_s0Vr','D0Vr','gamma_aN','gamma_sN','deltaN','gamma_aV','gamma_sV','deltaV','SNf','I_aNf','I_sNf','R_aNf','R_sNf','SVf','I_aVf','I_sVf','R_aVf','R_sVf','SVrf','I_aVrf','I_sVrf','R_aVrf','R_sVrf','DNf','DVf','DVrf','Df']

means = np.zeros([1,len(params)]) # +1 para agregar S0f

for j in range(len(params)-25):
    param = fit[params[j]]
    means[0,j] = np.mean(param)

y_pred2 = fit.extract(['y_pred'])['y_pred']
pred = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)

pd.DataFrame(pred[1]).to_csv(pathOutput+'/pred_0500.csv',header=['SN','I_aN','I_sN','R_aN','R_sN','DN','Ac','SV','I_aV','I_sV','R_aV','R_sV','DV','D','SVr','I_aVr','I_sVr','R_aVr','R_sVr','DVr'])
pd.DataFrame(pred[0]).to_csv(pathOutput+'/pred_0025.csv',header=['SN','I_aN','I_sN','R_aN','R_sN','DN','Ac','SV','I_aV','I_sV','R_aV','R_sV','DV','D','SVr','I_aVr','I_sVr','R_aVr','R_sVr','DVr'])
pd.DataFrame(pred[2]).to_csv(pathOutput+'/pred_0975.csv',header=['SN','I_aN','I_sN','R_aN','R_sN','DN','Ac','SV','I_aV','I_sV','R_aV','R_sV','DV','D','SVr','I_aVr','I_sVr','R_aVr','R_sVr','DVr'])

SNf = pred[1,-1,0] # S0 vacunados FINAL PARA EL SIGUIENTE MES
I_aNf = pred[1,-1,1]
I_sNf = pred[1,-1,2]
R_aNf = pred[1,-1,3]
R_sNf = pred[1,-1,4]
DNf = pred[1,-1,5] # S0 vacunados FINAL PARA EL SIGUIENTE MES
SVf = pred[1,-1,7] # S0 vacunados FINAL PARA EL SIGUIENTE MES
I_aVf = pred[1,-1,8]
I_sVf = pred[1,-1,9]
R_aVf = pred[1,-1,10]
R_sVf = pred[1,-1,11]
DVf = pred[1,-1,12] # S0 vacunados FINAL PARA EL SIGUIENTE MES
Df = pred[1,-1,13] # S0 vacunados FINAL PARA EL SIGUIENTE MES
SVrf = pred[1,-1,14] # S0 vacunados FINAL PARA EL SIGUIENTE MES
I_aVrf = pred[1,-1,15]
I_sVrf = pred[1,-1,16]
R_aVrf = pred[1,-1,17]
R_sVrf = pred[1,-1,18]
DVrf = pred[1,-1,19] # S0 vacunados FINAL PARA EL SIGUIENTE MES


means[0,-25] = gamma_aN
means[0,-24] = gamma_sN
means[0,-23] = deltaN
means[0,-22] = gamma_aV
means[0,-21] = gamma_sV
means[0,-20] = deltaV
means[0,-19] = SNf
means[0,-18] = I_aNf
means[0,-17] = I_sNf
means[0,-16] = R_aNf
means[0,-15] = R_sNf
means[0,-14] = SVf
means[0,-13] = I_aVf
means[0,-12] = I_sVf
means[0,-11] = R_aVf
means[0,-10] = R_sVf
means[0,-9] = SVrf
means[0,-8] = I_aVrf
means[0,-7] = I_sVrf
means[0,-6] = R_aVrf
means[0,-5] = R_sVrf
means[0,-4] = DNf
means[0,-3] = DVf
means[0,-2] = DVrf
means[0,-1] = Df


ox=pd.DataFrame(means,columns=params)
ox2 = ox.T
ox2.to_csv(pathOutput+'/tasas_means.csv')


print('Plotting prediction')
T_pred1=T_pred

plot_odepiso1(T, y, T_pred1, y_pred2,T_val,data_val)
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/pred_1.png', bbox_inches='tight')
plt.show()
plot_odepiso2(T, y, T_pred1, y_pred2,T_val,data_val)
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/pred_2.png', bbox_inches='tight')
plt.show()
plot_odepiso3(T, y, T_pred1, y_pred2,T_val,data_val)
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/pred_3.png', bbox_inches='tight')
plt.show()
#
dataplot=pd.date_range(start=t0,end=tf)
dataplot2=pd.date_range(start=tf,end=tval)
dataplot3=pd.date_range(start=t0,end=tpred)


#Acumulados
pred2 = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
k=6
plt.figure()
plt.title('Casos totales')
plt.plot(dataplot, y[:,k], color='k', marker='o', linestyle='none', ms=3)
plt.plot(dataplot2,data_val['Cases'],color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot3, pred2[0,:,k], color=colors[k], ls='--')
plt.plot(dataplot3, pred2[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
plt.plot(dataplot3, pred2[2,:,k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/Casos.png')
plt.show()



#Muertos

pred2 = np.quantile(y_pred2, [0.025, 0.5, 0.975], axis=0)
k=13
plt.figure()
plt.title('Fallecidos')
plt.plot(dataplot, y[:,k], color='k', marker='o', linestyle='none', ms=3)
plt.plot(dataplot2,data_val['Deaths'],color='r',linestyle='none',marker='o',ms=3)
plt.plot(dataplot3, pred2[0,:,k], color=colors[k], ls='--')
plt.plot(dataplot3, pred2[1,:,k], color=colors[k], ls='-', label='$%s$'%names[k])
plt.plot(dataplot3, pred2[2,:,k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/Fallecidos.png')
plt.show()

