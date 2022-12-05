# Importanción de modulos
from datetime import datetime
import os
import hashlib
import pickle
import re
import numpy as np
import scipy as sp
import pandas as pd
import stan as pystan # Original: import pystan
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment if you want to use multiprocessing
#import multiprocessing
#multiprocessing.set_start_method("fork", force=True)

# Descomentar
#jobid = sys.argv[1]
pathOutput = "Results"
#os.mkdir(pathOutput)


#pathOutput = "output/oneLayer/" + jobid + "/"

#pathStan = "stanModels/oneLayer/"
#sns.set()
#os.chdir('./')

# stanmod = sys.argv[2]
# S0_anterior = float(sys.argv[3])
#print(sys.argv[0])
#mes = sys.argv[4]
mes = "diciembre"
# I_a0_anterior = float(sys.argv[5])
# I_s0_anterior = float(sys.argv[6])
# R_a0_anterior = float(sys.argv[7])

# Nuevo
pathStan = "stan/oneLayer.stan"


# # compile the model (uncomment if needed)
#sm = pystan.build(file=pathStan, )

# # save the model to prevent compilation the next time
# # comment if already compiled
#with open(pathStan+'oneLayer.pkl', 'wb') as f:
#     pickle.dump(sm, f)

# load the model if already compiled and regitered
#with open(pathStan+'oneLayer.pkl', 'rb') as f:
#    sm = pickle.load(f)

# Datos para el modelo
# fechas para entrenar

# DICIEMBRE
if mes == "diciembre":
  t0 = '2020-12-01'
  tf = '2021-02-28'
  tval = '2021-03-01'
  tpred = '2021-03-07'
elif mes == "enero":
# ENERO
  t0 = '2021-01-01'
  tf = '2021-01-31'
  tval = '2021-02-01'
  tpred = '2021-02-07'
elif mes == "febrero":
# FEBRERO
  t0 = '2021-02-01'
  tf = '2021-02-28'
  tval = '2021-03-01'
  tpred = '2021-03-07'


# DICIEMBRE+ENERO
#t0 = '2020-12-01'
#tf = '2021-01-31'
#tval = '2021-02-01'
#tpred = '2021-02-07'

# cargar el archivo de datos
data = pd.read_csv('input/chileconvacunas.csv', index_col=0)
#data = pd.read_csv('input/chileconvacunas_media_movil.csv', index_col=0)
#data = pd.read_csv('input/casesAndDeathsChile.csv', index_col=0)
data.index = pd.to_datetime(data.index, dayfirst=False)
data_tr = data[t0:tf]
data_val = data[tf:tval]
data_pred = data[t0:tpred]

# tiempo de entreno
T = np.arange(0, len(data_tr))
# tiempo de validacióin
T_val = np.arange(len(data_tr), len(data_tr)+len(data_val))
# tiempo de predicción
T_pred = np.arange(0, len(data_pred))

# población de chile
N0 = 19458310
# %pob contagiada y aislada
mu = 0.5
# grado de confinamiento
ut = 0.5
q_a = 1/480
q_s = 1/480
p = 0.3
I0datos = 9089
#subreporte = 0.24
# DESCOMENTAR si se usa S0 fijo. Además, agregar en stan_data
# diciembre
#S0 = (1-13.48066419507794/100)*N0
# enero
#S0 = (1-14.457022055456996/100)*N0

# Vector con los datos
y = np.zeros((len(T), 7))
#y[:, 5] = data_tr['Fallecidos']
#y[:, 6] = data_tr['Casos totales']/subreporte
y[:, 5] = data_tr['Deaths']
y[:, 6] = data_tr['Cases']
#y[:, 6] = data_tr['Cases']/subreporte

print(f"y[5]: {y[:,5]}")
print(f"y[6]: {y[:,6]}")

# Entrega a stan
if mes == "diciembre":
	stan_data = {
	      'N': len(T),
	      'T': T,
	      'N_pred': len(T_pred),
	      'T_pred': T_pred,
	      'N0': N0,
	      'mu': mu,
	      'ut': ut,
	      'q_a': q_a,
	      'q_s': q_s,
	      'p': p,
	      'yD': y[:, 5],
	      'yIs': y[:, 6],
	      'I0datos': I0datos,
	}
else:
        stan_data = {
              'N': len(T),
              'T': T,
              'N_pred': len(T_pred),
              'T_pred': T_pred,
              'N0': N0,
              'mu': mu,
              'ut': ut,
              'q_a': q_a,
              'q_s': q_s,
              'p': p,
              'yD': y[:, 5],
              'yIs': y[:, 6],
	      'S0_anterior': S0_anterior,
              'I_a0_anterior': I_a0_anterior,
              'I_s0_anterior': I_s0_anterior,
              'R_a0_anterior': R_a0_anterior,

        }

#      'S0_anterior': S0_anterior,

# Condiciones para el modelo y correrlo
# warmup el default que se utilizó antes era iter/2
# chains debería ser igual al número de cores

code = """ functions {
    real[] dz_dt(real t, real[] z, real[] theta,real[] x_r, int[] x_i){

        // infered parameters
        real gamma_a= theta[1];
        real gamma_s= theta[2];
        real delta  = theta[3];
        real beta_a = theta[4];
        real beta_s = theta[5];
	real subr   = theta[6];

        // fixed parameters
        real q_a    = x_r[1];
        real q_s    = x_r[2];
        real ut     = x_r[3];
        real mu     = x_r[4];
        real p      = x_r[5];


        // variables of the ode
        real S   = z[1];
        real I_a = z[2];
        real I_s = z[3];
        real R_a = z[4];
        real R_s = z[5];
        real D   = z[6];
        real Ac  = z[7];
        real NT  = S+I_a+I_s+R_a+R_s;
        real Lambda = ((1-ut)*beta_a*I_a+(1-mu)*(beta_s*I_s))*S/NT;

        return {
             - Lambda + q_a*R_a+q_s*R_s,
            p*Lambda - gamma_a*I_a,
            (1-p) * Lambda - (gamma_s+delta)*I_s,
            gamma_a * I_a - q_a*R_a,
            gamma_s * I_s - q_s*R_s,
            delta * I_s,
            subr*(1-p) * Lambda
	    };
        }
    }

data {
    int<lower=0> N; // cantidad de días.
    real T[N]; //tiempos

    int<lower=0> N_pred;
    real T_pred[N_pred];

    //datos conocidos
    real yIs[N]; //infectados sintomáticos acumulados
    real yD[N]; //muertos

    // parametros constantes
    real<lower=0> N0; //población de Chile
    real<lower=0> mu;
    real<lower=0> ut;
    real<lower=0> q_a;
    real<lower=0> q_s;
    real<lower=0> p;// probability of being asintomatic

    real<lower=0> I0datos; // guest inicial para I0
    }

transformed data { //se usan para resolver la edo
    real x_r[5] = {q_a, q_s, ut, mu,p};
    int x_i[0];
    real rel_tol = 1e-6;
    real abs_tol = 1e-6;
    real max_num_steps = 1e4;
}

//parametros que van a ser sampleados por stan, los que vamos a encontrar
parameters {
real<lower=0,upper=N0> S0;
real<lower=0,upper=N0-S0> I_a0;
real<lower=0,upper=N0-S0-I_a0> I_s0;
real<lower=0,upper=N0-S0-I_s0-I_a0> R_a0;
real<lower=0,upper=N0-S0-I_s0-I_a0-R_a0> D0;

real<lower=0,upper=N0> Ac0;

real<lower=0.01,upper=1.0> beta_s;
real<lower=0.01,upper=1.0> beta_a;
real<lower=1/30.0,upper=1/2.0> gamma_a;
real<lower=1/30.0,upper=gamma_a> gamma_s;
real<lower=0.001,upper=0.50> delta;
#real<lower=0.6,upper=0.7> subr;
real<lower=0.7,upper=0.8> subr;

real<lower=0,upper=1> sigma; //se usa en la estamación de las cond iniciales
}

transformed parameters {
real theta[6] = {gamma_a, gamma_s, delta, beta_a, beta_s,subr};
//real I0=I_a0+I_s0;
real R_s0=N0-S0-I_s0-I_a0-R_a0-D0;
real z[N,7];
z[1]={S0,I_a0,I_s0,R_a0,R_s0,D0,Ac0};
z[2:,] = integrate_ode_rk45(dz_dt, z[1], 0, T[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);

}
model {
// priors
 //initial conditions
 S0/N0 ~ normal(1-14.702281005400776/100,sigma);
 I_a0 ~ lognormal(log(p*I0datos/0.44),sigma);
 I_s0 ~ lognormal(log((1-p)*I0datos/0.44),sigma);
 R_a0/N0 ~ beta(1,4);
 Ac0 ~ lognormal(log(yIs[1]), sigma);
 D0 ~ lognormal(log(yD[1]), sigma);

 // subreporte
 subr ~ beta(2,2);

 //rate
  beta_s ~ normal(0.1, 0.50);
  beta_a ~ normal(0.1, 0.50);
  delta ~ normal(0.1, 0.50);
  1/gamma_a ~ normal(14.0, 15.0);
  1/gamma_s ~ normal(14.0, 15.0);
  sigma ~ cauchy(0.0, 1.0); // 0, 0.2
  yD  ~ lognormal(log(z[,6]), sigma);
  yIs ~ lognormal(log(z[,7]), sigma);
  //yD  ~ normal(z[,6], sigma);
  //yIs ~ normal(z[,7], sigma);
}

generated quantities {
    real y_pred[N_pred,7];
    y_pred[1,] = {S0,I_a0,I_s0,R_a0,R_s0,D0,Ac0};
    y_pred[2:,] = integrate_ode_rk45(dz_dt, y_pred[1], 0, T_pred[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);
}

"""

sm = pystan.build(code, data = stan_data)

fit = sm.sample(num_chains=2, num_samples=50)
#fit = sm.sample(num_chains=10, num_samples=200, check_hmc_diagnostics=False, control={'adapt_delta':0.96, 'max_treedepth': 14})

#pystan.check_hmc_diagnostics(fit, verbose=True)
summary = fit.stansummary()
z_last = re.findall("z\[%d,.*" % (len(T),), summary)
summary = re.sub("(y_pred|z|theta)\[.*\n?", "", summary)
print(summary)

print()
for z in z_last:
    print(z)
print()

##
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
names = ['S(t)', 'I_a(t)', 'I_s(t)', 'R_a(t)', 'R_s(t)', 'D(t)', 'I_ac(t)']
# ploteo solución completa


def plot_ode(t_obs, obs, t_pred, param, t_val, val):
    pred = np.quantile(param, [0.025, 0.5, 0.975], axis=0)
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(20, 20), squeeze=False)
    fig.suptitle(f'MCMC fit for Región Metropolitana {mes}')
    aj = ['Deaths', 'Cases']
    #aj = ['Fallecidos','Casos totales']
    con = 0
    for k in range(7):
        ax = axs[int(k/3), k % 3]
        if k == 5:
            ax.plot(t_obs, obs[:, k], color=colors[k],
                    marker='o', linestyle='none', ms=3)
            ax.plot(t_val, val[aj[con]], color='r',
                    linestyle='none', marker='o', ms=3)
            con = con+1
        elif k == 6:
            ax.plot(t_obs, obs[:, k], color=colors[k],
                    marker='o', linestyle='none', ms=3)
#            ax.plot(t_val, val[aj[con]]/subreporte, color='r',
#                    linestyle='none', marker='o', ms=3)
            ax.plot(t_val, val[aj[con]], color='r',
                    linestyle='none', marker='o', ms=3)
            con = con+1
        ax.plot(t_pred, pred[0, :, k], color=colors[k], ls='--')
        ax.plot(t_pred, pred[1, :, k], color=colors[k],
                ls='-', label='$%s$' % names[k])
        ax.plot(t_pred, pred[2, :, k], color=colors[k], ls='--')
        ax.legend()


print('Plotting prediction')
y_pred1 = fit.extract(['y_pred'])['y_pred']
pred = np.quantile(y_pred1, [0.025, 0.5, 0.975], axis=0)

T_pred1 = T_pred
plot_ode(T, y, T_pred1, y_pred1, T_val, data_val)
plt.savefig(pathOutput+'/pred4.png', bbox_inches='tight')
plt.xticks(rotation=45)
plt.show()

params = ['gamma_a', 'gamma_s', 'delta', 'beta_a', 'beta_s','subr','S0','I_s0','I_a0','R_a0','R_s0','D0','S0f','Iaf','Isf','Raf','Rsf','D0f']
means = np.zeros([1,len(params)]) # +1 para agregar S0f
stds = np.zeros([1,len(params)])

for j in range(len(params)-6):
    param = fit[params[j]]
    means[0,j] = np.mean(param)
#    stds[j] = np.std(param)


S0f = pred[1,-1,0] # S0 FINAL PARA EL SIGUIENTE MES
Iaf = pred[1,-1,1] # Ia FINAL PARA EL SIGUIENTE MES
Isf = pred[1,-1,2] # Is FINAL PARA EL SIGUIENTE MES
Raf = pred[1,-1,3] # Ra FINAL PARA EL SIGUIENTE MES
Rsf = pred[1,-1,4] # Rs FINAL PARA EL SIGUIENTE MES 
D0f = pred[1,-1,5] # D0 FINAL PARA EL SIGUIENTE MES

means[0,-6] = S0f
means[0,-5] = Iaf
means[0,-4] = Isf
means[0,-3] = Raf
means[0,-2] = Rsf
means[0,-1] = D0f

ox=pd.DataFrame(means,columns=params)
ox2 = ox.T
ox2.to_csv(pathOutput+'/tasas_means.csv')
#ox=pd.DataFrame(stds,columns=params)
#ox.T.to_csv(pathOutput+'/tasas_stds.csv')


##

dataplot = pd.date_range(start=t0, end=tf)
dataplot2 = pd.date_range(start=tf, end=tval)
dataplot3 = pd.date_range(start=t0, end=tpred)

# D
pred = np.quantile(y_pred1, [0.025, 0.5, 0.975], axis=0)
k = 5
plt.figure()
plt.title('Fallecidos')
plt.plot(dataplot, y[:, k], color='k', marker='o', linestyle='none', ms=3)
plt.plot(dataplot2, data_val['Deaths'],
         color='r', linestyle='none', marker='o', ms=3)
plt.plot(dataplot3, pred[0, :, k], color=colors[k], ls='--')
plt.plot(dataplot3, pred[1, :, k], color=colors[k],
         ls='-', label='$%s$' % names[k])
plt.plot(dataplot3, pred[2, :, k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/muertes4.png')
plt.show()

# Acumulados
pred = np.quantile(y_pred1, [0.025, 0.5, 0.975], axis=0)
k = 6
plt.figure()
plt.title('Casos totales')
plt.plot(dataplot, y[:, k], color='k', marker='o', linestyle='none', ms=3)
#plt.plot(dataplot2, data_val['Cases']/subreporte,
#         color='r', linestyle='none', marker='o', ms=3)
plt.plot(dataplot2, data_val['Cases'],
         color='r', linestyle='none', marker='o', ms=3)
plt.plot(dataplot3, pred[0, :, k], color=colors[k], ls='--')
plt.plot(dataplot3, pred[1, :, k], color=colors[k],
         ls='-', label='$%s$' % names[k])
plt.plot(dataplot3, pred[2, :, k], color=colors[k], ls='--')
plt.xticks(rotation=45)
plt.savefig(pathOutput+'/Casos totales4.png')
plt.show()
