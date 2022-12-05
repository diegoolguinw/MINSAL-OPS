# Libraries
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Seaborn theme
sns.set_theme()

sns.set(font_scale=1.4)

# Uncomment if necessary to change the work directory
#import os
#path = ""
#os.chdir(path)

# FIRST PART #
t0 = '2021-08-16' # Initial time
# Read .csv of vaccinated people data
data = pd.read_csv('input/chileconvacunasvf.csv', sep=';', index_col=0)
# The DataFrame index as dates
data.index = pd.to_datetime(data.index, dayfirst=True)

# Final time to interpolate
tintp = '2022-1-15'
# Arrays of data from initial time to final interpolation time
data_intp = data[t0:tintp]
T_intp = np.arange(0, len(data_intp))

# Arrays of data keys
ye    = data_intp['Efectividad']
ye2   = data_intp['Efe refuerzo']
yVd0  = data_intp['Vacunados']
yVrd0 = data_intp['refuerzo']
yVd   = np.zeros_like(yVd0)
yVrd  = np.zeros_like(yVrd0)

# Adjust interpolation time to reach to final time
for i in range(len(yVd)-1):
    yVd[i]=yVd0[i+1]-yVd0[i]
    yVrd[i]=yVrd0[i+1]-yVrd0[i]
yVd[-1]  = yVd[-2]
yVrd[-1] = yVrd[-2]

# Interpolation
ft  = interp1d(T_intp,ye)
ft2 = interp1d(T_intp,ye2)
v   = interp1d(T_intp,yVd)
vr  = interp1d(T_intp,yVrd)

# ODE parameters
ut    = 0.5
mu    = 0.5
pN    = 0.3
q_aN  = 1/480
q_sN  = 1/480
pV    = 0.7
q_aV  = 1/480
q_sV  = 1/480
pVr   = 0.7
q_aVr = 1/480
q_sVr = 1/480
N0    = 19458310

# ODE function
def dzdt(z,t):
    # Parameters
    SN    = z[0]
    I_aN  = z[1]
    I_sN  = z[2]
    R_aN  = z[3]
    R_sN  = z[4]
    DN    = z[5]
    Ac    = z[6]
    SV    = z[7]
    I_aV  = z[8]
    I_sV  = z[9]
    R_aV  = z[10]
    R_sV  = z[11]
    DV    = z[12]
    D     = z[13]
    SVr   = z[14]
    I_aVr = z[15]
    I_sVr = z[16]
    R_aVr = z[17]
    R_sVr = z[18]
    DVr   = z[19]

    # Total population
    NT  = (SN + I_aN + I_sN + R_aN + R_sN) +(SV + I_aV + I_sV + R_aV + R_sV) +(SVr + I_aVr + I_sVr + R_aVr + R_sVr);

    # New infeccted not vaccinated
    LambdaN = ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SN / NT;

    # New infeccted vaccinated
    LambdaV = (1-ft(t)) *  ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SV / NT;

    # New infeccted with booster vaccine
    LambdaVr = (1-ft2(t)) * ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SVr / NT;

    # New vaccinated with primary scheme
    # New vaccinated in the compartiment: susceptible vaccinated
    vSVN = v(t) * SN / (SN + R_aN + R_sN);
    # New vaccinated in the compartiment: recovered asymptomatic
    vRNa = v(t) * R_aN / (SN + R_aN + R_sN);
    # New vaccinated in the compartiment: recovered symptomatic
    vRNs = v(t) * R_sN / (SN + R_aN + R_sN);

    # New boosted vaccinated
    # New vaccinated in the compartiment: susceptible vaccinated
    vSVNr = vr(t) * SV / (SV + R_aV + R_sV);
    # New vaccinated in the compartiment: recovered asymptomatic
    vRNar = vr(t) * R_aV / (SV + R_aV + R_sV);
    # New vaccinated in the compartiment: recovered symptomatic
    vRNsr = vr(t) * R_sV / (SV + R_aV + R_sV);


    solution = [
        # First fundamental unit
        -LambdaN+q_aN*R_aN+q_sN*R_sN-vSVN, # Susceptible
        pN*LambdaN-gamma_aN*I_aN,          # Asymptomatic infected
        (1-pN)*LambdaN-(gamma_sN+deltaN)*I_sN, # Symptomatic infected
        gamma_aN*I_aN-q_aN*R_aN-vRNa,          # Asymptomatic recovered
        gamma_sN*I_sN-q_sN*R_sN-vRNs,          # Symptomatic recovered
        deltaN*I_sN,                           # Deceased

        # Acumulated symptomatic cases
        subr*((1-pN)*LambdaN+(1-pV)*LambdaV+(1-pVr)*LambdaVr), 

        # Second fundamental unit
        -LambdaV+q_aV*R_aV+q_sV*R_sV+v(t)-vSVNr, # Susceptible
        pV*LambdaV-gamma_aV*I_aV,                # Asymptomatic infected
        (1-pV)*LambdaV-(gamma_sV+deltaV)*I_sV,   # Symptomatic infected
        gamma_aV*I_aV-q_aV*R_aV-vRNar,           # Asymptomatic recovered
        gamma_sV*I_sV-q_sV*R_sV-vRNsr,           # Symptomatic recovered
        deltaV*I_sV,                             # Deceased

        # Acumulated deceased
        deltaN*I_sN+deltaV*I_sV+deltaVr*I_sVr, 

        # Third fundamental unit
        -LambdaVr+q_aVr*R_aVr+q_sVr*R_sVr+vr(t),    # Susceptible
        pVr*LambdaVr-gamma_aVr*I_aVr,               # Asymptomatic infected
        (1-pVr)*LambdaVr-(gamma_sVr+deltaVr)*I_sVr, # Symptomatic infected
        gamma_aVr*I_aVr-q_aVr*R_aVr,                # Asymptomatic recovered
        gamma_sVr*I_sVr-q_sVr*R_sVr,                # Symptomatic recovered
        deltaVr*I_sVr                               # Deceased
    ]

    return np.array(solution)

# ODE without third fundamental unit (no boost vaccine)
def dzdt_NoBoost(z,t):
    # ODE parameters
    SN   = z[0]
    I_aN = z[1]
    I_sN = z[2]
    R_aN = z[3]
    R_sN = z[4]
    DN   = z[5]
    Ac   = z[6]
    SV   = z[7]
    I_aV = z[8]
    I_sV = z[9]
    R_aV = z[10]
    R_sV = z[11]
    DV   = z[12]
    D    = z[13]

    # Total population
    NT  = (SN + I_aN + I_sN + R_aN + R_sN) + (SV + I_aV + I_sV + R_aV + R_sV);

    # New infeccted not vaccinated
    LambdaN = ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV)) * SN / NT;

    # New infeccted vaccinated
    LambdaV = (1-ft(t)) *  ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV)) * SV / NT;
    
    # New vaccinated in the compartiment: susceptible vaccinated
    vSVN = v(t) * SN / (SN + R_aN + R_sN);
    # New vaccinated in the compartiment: recovered asymptomatic
    vRNa = v(t) * R_aN / (SN + R_aN + R_sN);
    # New vaccinated in the compartiment: recovered symptomatic
    vRNs = v(t) * R_sN / (SN + R_aN + R_sN);

    solution = [
        # First fundamental unit
        -LambdaN+q_aN*R_aN+q_sN*R_sN-vSVN,     # Susceptible
        pN*LambdaN-gamma_aN*I_aN,              # Asymptomatic infected
        (1-pN)*LambdaN-(gamma_sN+deltaN)*I_sN, # Symptomatic infected
        gamma_aN*I_aN-q_aN*R_aN-vRNa,          # Asymptomatic recovered
        gamma_sN*I_sN-q_sN*R_sN-vRNs,          # Symptomatic recovered
        deltaN*I_sN,                           # Deceased

        # Acumulated symptomatic cases
        subr*((1-pN)*LambdaN+(1-pV)*LambdaV),

        # Second fundamental unit
        -LambdaV+q_aV*R_aV+q_sV*R_sV+v(t),      # Susceptible
        pV*LambdaV-gamma_aV*I_aV,               # Asymptomatic infected
        (1-pV)*LambdaV-(gamma_sV+deltaV)*I_sV,  # Symptomatic infected
        gamma_aV*I_aV-q_aV*R_aV,                # Asymptomatic recovered
        gamma_sV*I_sV-q_sV*R_sV,                # Symptomatic recovered
        deltaV*I_sV,                            # Deceased
        deltaN*I_sN+deltaV*I_sV                 # Acumulated deceased
    ]

    return np.array(solution) 

#################
# Solve ODE
#################
# Read more date

# Rates
rates = pd.read_csv('soluciones/15ago_1oct/tasas_means.csv',index_col=0)
rates = rates['valores']

# Solutions
sol_b1 = pd.read_csv('soluciones/15ago_1oct/pred_0500.csv',index_col=0)
sol_conf1 = pd.read_csv('soluciones/15ago_1oct/pred_0025.csv',index_col=0)
sol_conf2 = pd.read_csv('soluciones/15ago_1oct/pred_0975.csv',index_col=0)

# Initial condition for three fundamental units model
z0_b1 = np.array(list(sol_b1.iloc[0]))
z0_conf1_b1 = np.array(list(sol_conf1.iloc[0]))
z0_conf2_b1 = np.array(list(sol_conf2.iloc[0]))

# Initial condition for two fundamental units model
z02_b1 = np.array(list(sol_b1.iloc[0][0:14]))
z02_conf1_b1 = np.array(list(sol_conf1.iloc[0][0:14]))
z02_conf2_b1 = np.array(list(sol_conf2.iloc[0][0:14]))

###############
# First period
###############
t0_b1 = '2021-08-16' # Initial time first period
tf_b1 = '2021-09-30' # Final time first perido

# Data for the first period
data_tr_b1 = data[t0_b1:tf_b1]
T_b1 = np.arange(0, len(data_tr_b1))

# Unpack rates

# First fundamental unit
gamma_aN = rates['gamma_aN']
gamma_sN = rates['gamma_sN']
deltaN   = rates['deltaN']
beta_aN  = rates['beta_aN']
beta_sN  = rates['beta_sN']

# Second fundamental unit
gamma_aV = rates['gamma_aV']
gamma_sV = rates['gamma_sV']
deltaV   = rates['deltaV']
beta_aV  = rates['beta_aV']
beta_sV  = rates['beta_sV']

# Third fundamental unit
gamma_aVr = rates['gamma_aVr']
gamma_sVr = rates['gamma_sVr']
deltaVr   = rates['deltaVr']
beta_aVr  = rates['beta_aVr']
beta_sVr  = rates['beta_sVr']

# Subreport
subr = rates['subr']

# Adjust initial conditions
z02_b1[7] = z02_b1[7] + 260246
z02_conf1_b1[7] = z02_conf1_b1[7] + 260246
z02_conf2_b1[7] = z02_conf2_b1[7] + 260246

# Solution for the first period with three fundamental units
z_b1  = odeint(dzdt     , z0_b1 , T_b1)
z_conf1_b1 = odeint(dzdt, z0_conf1_b1 , T_b1)
z_conf2_b1 = odeint(dzdt, z0_conf2_b1 , T_b1)
# Solution for the first period with two fundamental units
z_NB_b1 = odeint(dzdt_NoBoost, z02_b1, T_b1)
z_NB_conf1_b1 = odeint(dzdt, z0_conf1_b1 , T_b1)
z_NB_conf2_b1 = odeint(dzdt, z0_conf2_b1 , T_b1)

################
# Second period
################

# Rates for the second period
rates2 = pd.read_csv('soluciones/1oct_31dec/tasas_means.csv',index_col=0)
rates2 = rates2['valores']

# Solutions for the second period
sol_b2 = pd.read_csv('soluciones/1oct_31dec/pred_0500.csv',index_col=0)
sol_conf1_b2 = pd.read_csv('soluciones/1oct_31dec/pred_0025.csv',index_col=0)
sol_conf2_b2 = pd.read_csv('soluciones/1oct_31dec/pred_0975.csv',index_col=0)

# Initial condition for the second period with three fundamental units
z0_b2  = np.array(list(sol_b2.iloc[0]))
z0_conf1_b2 = np.array(list(sol_conf1_b2.iloc[0]))
z0_conf2_b2 = np.array(list(sol_conf2_b2.iloc[0]))

# Initial condition for the second period with two fundamental units
z02_b2 = np.array(list(sol_b2.iloc[0][0:14]))
z02_conf1_b2 = np.array(list(sol_conf1_b2.iloc[0][0:14]))
z02_conf2_b2 = np.array(list(sol_conf2_b2.iloc[0][0:14]))

# Initial and final time for the second period
t0_b2 = '2021-10-01'
tf_b2 = '2021-12-31'
# Second period data
data_tr_b2 = data[t0_b2:tf_b2]
T_b2 = np.arange(0, len(data_tr_b2))

# Unpack rates
# First fundamental unit
gamma_aN = rates2['gamma_aN']
gamma_sN = rates2['gamma_sN']
deltaN   = rates2['deltaN']
beta_aN  = rates2['beta_aN']
beta_sN  = rates2['beta_sN']

# Second fundamental unit
gamma_aV = rates2['gamma_aV']
gamma_sV = rates2['gamma_sV']
deltaV   = rates2['deltaV']
beta_aV  = rates2['beta_aV']
beta_sV  = rates2['beta_sV']

# Third fundamental unit
gamma_aVr = rates2['gamma_aVr']
gamma_sVr = rates2['gamma_sVr']
deltaVr   = rates2['deltaVr']
beta_aVr  = rates2['beta_aVr']
beta_sVr  = rates2['beta_sVr']

# Subreport
subr=rates2['subr']

z02_b2[7:13] = z02_b2[7:13] + z0_b2[14:20]
z02_conf1_b2[7:13] = z02_conf1_b2[7:13] + z0_conf1_b2[14:20]
z02_conf2_b2[7:13] = z02_conf2_b2[7:13] + z0_conf2_b2[14:20]

# Solution given by the model for the second period with 3 fundamental units
z_b2  = odeint(dzdt,z0_b2,T_b2)
z_conf1_b2 = odeint(dzdt, z0_conf1_b2, T_b2)
z_conf2_b2 = odeint(dzdt, z0_conf2_b2, T_b2)
# Solution given by the model for the second period with 2 fundamental units
z_NB_b2 = odeint(dzdt_NoBoost,z02_b2,T_b2)
z_NB_conf1_b2 = odeint(dzdt_NoBoost, z02_conf1_b2, T_b2)
z_NB_conf2_b2 = odeint(dzdt_NoBoost, z02_conf2_b2, T_b2)


####
# Daily cases
####

new_cases_B1_dat = np.array([data_tr_b1['Cases'][i] - data_tr_b1['Cases'][i-1] for i in range(1,len(data_tr_b1['Cases']))])
new_cases_B1_no_boost = np.array([z_NB_b1[:,6][i] - z_NB_b1[:,6][i-1] for i in range(1,len(z_NB_b1[:,6]))])
new_cases_B1_mod = np.array([z_b1[:,6][i] - z_b1[:,6][i-1] for i in range(1,len(z_b1[:,6]))])
new_cases_B1_CI95_1 = np.array([z_conf1_b1[:,6][i] - z_conf1_b1[:,6][i-1] for i in range(1,len(z_conf1_b1[:,6]))])
new_cases_B1_CI95_2 = np.array([z_conf2_b1[:,6][i] - z_conf2_b1[:,6][i-1] for i in range(1,len(z_conf2_b1[:,6]))])
new_cases_NB_B1_CI95_1 = np.array([z_NB_conf1_b1[:,6][i] - z_NB_conf1_b1[:,6][i-1] for i in range(1,len(z_NB_conf1_b1[:,6]))])
new_cases_NB_B1_CI95_2 = np.array([z_NB_conf2_b1[:,6][i] - z_NB_conf2_b1[:,6][i-1] for i in range(1,len(z_NB_conf2_b1[:,6]))])

new_cases_B2_dat = np.array([data_tr_b2['Cases'][i] - data_tr_b2['Cases'][i-1] for i in range(1,len(data_tr_b2['Cases']))])
new_cases_B2_no_boost = np.array([z_NB_b2[:,6][i] - z_NB_b2[:,6][i-1] for i in range(1,len(z_NB_b2[:,6]))])
new_cases_B2_mod = np.array([z_b2[:,6][i] - z_b2[:,6][i-1] for i in range(1,len(z_b2[:,6]))])
new_cases_B2_CI95_1 = np.array([z_conf1_b2[:,6][i] - z_conf1_b2[:,6][i-1] for i in range(1,len(z_conf1_b2[:,6]))])
new_cases_B2_CI95_2 = np.array([z_conf2_b2[:,6][i] - z_conf2_b2[:,6][i-1] for i in range(1,len(z_conf2_b2[:,6]))])
new_cases_NB_B2_CI95_1 = np.array([z_NB_conf1_b2[:,6][i] - z_NB_conf1_b2[:,6][i-1] for i in range(1,len(z_NB_conf1_b2[:,6]))])
new_cases_NB_B2_CI95_2 = np.array([z_NB_conf2_b2[:,6][i] - z_NB_conf2_b2[:,6][i-1] for i in range(1,len(z_NB_conf2_b2[:,6]))])

####
# Daily deceases
####

new_deaths_B1_dat = np.array([data_tr_b1['Deaths'][i] - data_tr_b1['Deaths'][i-1] for i in range(1,len(data_tr_b1['Deaths']))])
new_deaths_B1_no_boost = np.array([z_NB_b1[:,13][i] - z_NB_b1[:,13][i-1] for i in range(1,len(z_NB_b1[:,13]))])
new_deaths_B1_mod = np.array([z_b1[:,13][i] - z_b1[:,13][i-1] for i in range(1,len(z_b1[:,13]))])
new_deaths_B1_CI95_1 = np.array([z_conf1_b1[:,13][i] - z_conf1_b1[:,13][i-1] for i in range(1,len(z_conf1_b1[:,13]))])
new_deaths_B1_CI95_2 = np.array([z_conf2_b1[:,13][i] - z_conf2_b1[:,13][i-1] for i in range(1,len(z_conf2_b1[:,13]))])
new_deaths_NB_B1_CI95_1 = np.array([z_NB_conf1_b1[:,13][i] - z_NB_conf1_b1[:,13][i-1] for i in range(1,len(z_NB_conf1_b1[:,13]))])
new_deaths_NB_B1_CI95_2 = np.array([z_NB_conf2_b1[:,13][i] - z_NB_conf2_b1[:,13][i-1] for i in range(1,len(z_NB_conf2_b1[:,13]))])

new_deaths_B2_dat = np.array([data_tr_b2['Deaths'][i] - data_tr_b2['Deaths'][i-1] for i in range(1,len(data_tr_b2['Deaths']))])
new_deaths_B2_no_boost = np.array([z_NB_b2[:,13][i] - z_NB_b2[:,13][i-1] for i in range(1,len(z_NB_b2[:,13]))])
new_deaths_B2_mod = np.array([z_b2[:,13][i] - z_b2[:,13][i-1] for i in range(1,len(z_b2[:,13]))])
new_deaths_B2_CI95_1 = np.array([z_conf1_b2[:,13][i] - z_conf1_b2[:,13][i-1] for i in range(1,len(z_conf1_b2[:,13]))])
new_deaths_B2_CI95_2 = np.array([z_conf2_b2[:,13][i] - z_conf2_b2[:,13][i-1] for i in range(1,len(z_conf2_b2[:,13]))])
new_deaths_NB_B2_CI95_1 = np.array([z_NB_conf1_b2[:,13][i] - z_NB_conf1_b2[:,13][i-1] for i in range(1,len(z_NB_conf1_b2[:,13]))])
new_deaths_NB_B2_CI95_2 = np.array([z_NB_conf2_b2[:,13][i] - z_NB_conf2_b2[:,13][i-1] for i in range(1,len(z_NB_conf2_b2[:,13]))])

####
# Total population
####

#### Complete

#######
# Metrics computation
#######


# Dates for each period
dates1 = data_tr_b1.index  # Dates for the first period
dates2 = data_tr_b2.index  # Dates for the second period

predCases = pd.DataFrame()
predCases['dates'] = np.concatenate((dates1, dates2))
predCases['cases'] = np.concatenate((z_NB_b1[:,6], z_NB_b2[:,6]))
predCases = predCases.set_index('dates')
predCases.to_csv('Results/casesNoVac.csv')

predDeaths = pd.DataFrame()
predDeaths['dates'] = np.concatenate((dates1, dates2))
predDeaths['deaths'] = np.concatenate((z_NB_b1[:,13], z_NB_b2[:,13]))
predDeaths = predDeaths.set_index('dates')
predDeaths.to_csv('Results/deathsNoVac.csv')


# Arrays of the acumulated cases difference
acum_dif1_mod = z_NB_b1[:,6] - z_b1[:,6]
acum_dif2_mod = z_NB_b2[:,6] - z_b2[:,6]
acum_dif1_dat = z_NB_b1[:,6] - data_tr_b1['Cases'] 
acum_dif2_dat = z_NB_b2[:,6] - data_tr_b2['Cases'] 

acum_dif1_mod_IC95_1 = z_NB_conf1_b1[:,6] - z_b1[:,6]
acum_dif1_mod_IC95_2 = z_NB_conf2_b1[:,6] - z_b1[:,6]
acum_dif2_mod_IC95_1 = z_NB_conf1_b2[:,6] - z_b2[:,6]
acum_dif2_mod_IC95_2 = z_NB_conf2_b2[:,6] - z_b2[:,6]
acum_dif1_dat_IC95_1 = z_NB_conf1_b1[:,6] - data_tr_b1['Cases'] 
acum_dif1_dat_IC95_2 = z_NB_conf2_b1[:,6] - data_tr_b1['Cases'] 
acum_dif2_dat_IC95_1 = z_NB_conf1_b2[:,6] - data_tr_b2['Cases'] 
acum_dif2_dat_IC95_2 = z_NB_conf2_b2[:,6] - data_tr_b2['Cases'] 

# Arrays of the deceased difference
decea_dif1_mod = z_NB_b1[:,13] - z_b1[:,13]
decea_dif2_mod = z_NB_b2[:,13] - z_b2[:,13]
decea_dif1_dat = z_NB_b1[:,13] - data_tr_b1['Deaths'] 
decea_dif2_dat = z_NB_b2[:,13] - data_tr_b2['Deaths'] 

decea_dif1_mod_IC95_1 = z_NB_conf1_b1[:,13] - z_b1[:,13]
decea_dif1_mod_IC95_2 = z_NB_conf2_b1[:,13] - z_b1[:,13]
decea_dif2_mod_IC95_1 = z_NB_conf1_b2[:,13] - z_b2[:,13]
decea_dif2_mod_IC95_2 = z_NB_conf2_b2[:,13] - z_b2[:,13]
decea_dif1_dat_IC95_1 = z_NB_conf1_b1[:,13] - data_tr_b1['Deaths'] 
decea_dif1_dat_IC95_2 = z_NB_conf2_b1[:,13] - data_tr_b1['Deaths'] 
decea_dif2_dat_IC95_1 = z_NB_conf1_b2[:,13] - data_tr_b2['Deaths'] 
decea_dif2_dat_IC95_2 = z_NB_conf2_b2[:,13] - data_tr_b2['Deaths'] 

# Arrays for the percentual acumulated cases difference
acum_dif1_mod_por = np.zeros(len(dates1))
acum_dif2_mod_por = np.zeros(len(dates2))
acum_dif1_dat_por = np.zeros(len(dates1))
acum_dif2_dat_por = np.zeros(len(dates2))

acum_dif1_mod_por_IC95_1 = np.zeros(len(dates1))
acum_dif1_mod_por_IC95_2 = np.zeros(len(dates1))
acum_dif2_mod_por_IC95_1 = np.zeros(len(dates2))
acum_dif2_mod_por_IC95_2 = np.zeros(len(dates2))
acum_dif1_dat_por_IC95_1 = np.zeros(len(dates1))
acum_dif1_dat_por_IC95_2 = np.zeros(len(dates1))
acum_dif2_dat_por_IC95_1 = np.zeros(len(dates2))
acum_dif2_dat_por_IC95_2 = np.zeros(len(dates2))

# Arrays for the percentual deceased difference
decea_dif1_mod_por = np.zeros(len(dates1))
decea_dif2_mod_por = np.zeros(len(dates2))
decea_dif1_dat_por = np.zeros(len(dates1))
decea_dif2_dat_por = np.zeros(len(dates2))

decea_dif1_mod_por_IC95_1 = np.zeros(len(dates1))
decea_dif1_mod_por_IC95_2 = np.zeros(len(dates1))
decea_dif2_mod_por_IC95_1 = np.zeros(len(dates2))
decea_dif2_mod_por_IC95_2 = np.zeros(len(dates2))
decea_dif1_dat_por_IC95_1 = np.zeros(len(dates1))
decea_dif1_dat_por_IC95_2 = np.zeros(len(dates1))
decea_dif2_dat_por_IC95_1 = np.zeros(len(dates2))
decea_dif2_dat_por_IC95_2 = np.zeros(len(dates2))

for i in range(1, len(dates1)-1):
    acum_dif1_mod_por[i] = 100*(new_cases_B1_no_boost[i] - new_cases_B1_mod[i])/(new_cases_B1_no_boost[i])
    acum_dif1_dat_por[i] = 100*(new_cases_B1_no_boost[i] - new_cases_B1_dat[i])/(new_cases_B1_no_boost[i])
    decea_dif1_mod_por[i] = 100*(new_deaths_B1_no_boost[i] - new_deaths_B1_mod[i])/(new_deaths_B1_no_boost[i])
    decea_dif1_dat_por[i] = 100*(new_deaths_B1_no_boost[i] - new_deaths_B1_dat[i])/(new_deaths_B1_no_boost[i])

    acum_dif1_mod_por_IC95_1[i] = 100*(new_cases_NB_B1_CI95_1[i] - new_cases_B1_mod[i])/(new_cases_NB_B1_CI95_1[i])
    acum_dif1_dat_por_IC95_1[i] = 100*(new_cases_NB_B1_CI95_1[i] - new_cases_B1_dat[i])/(new_cases_NB_B1_CI95_1[i])
    acum_dif1_mod_por_IC95_2[i] = 100*(new_cases_NB_B1_CI95_2[i] - new_cases_B1_mod[i])/(new_cases_NB_B1_CI95_2[i])
    acum_dif1_dat_por_IC95_2[i] = 100*(new_cases_NB_B1_CI95_2[i] - new_cases_B1_dat[i])/(new_cases_NB_B1_CI95_2[i])

    decea_dif1_mod_por_IC95_1[i] = 100*(new_deaths_NB_B1_CI95_1[i] - new_deaths_B1_mod[i])/(new_deaths_NB_B1_CI95_1[i])
    decea_dif1_dat_por_IC95_1[i] = 100*(new_deaths_NB_B1_CI95_1[i] - new_deaths_B1_dat[i])/(new_deaths_NB_B1_CI95_1[i])
    decea_dif1_mod_por_IC95_2[i] = 100*(new_deaths_NB_B1_CI95_2[i] - new_deaths_B1_mod[i])/(new_deaths_NB_B1_CI95_2[i])
    decea_dif1_dat_por_IC95_2[i] = 100*(new_deaths_NB_B1_CI95_2[i] - new_deaths_B1_dat[i])/(new_deaths_NB_B1_CI95_2[i])

for i in range(1, len(dates2)-1):
    acum_dif2_mod_por[i] = 100*(new_cases_B2_no_boost[i] - new_cases_B2_mod[i])/(new_cases_B2_no_boost[i])
    acum_dif2_dat_por[i] = 100*(new_cases_B2_no_boost[i] - new_cases_B2_dat[i])/(new_cases_B2_no_boost[i])
    decea_dif2_mod_por[i] = 100*(new_deaths_B2_no_boost[i] - new_deaths_B2_mod[i])/(new_deaths_B2_no_boost[i])
    decea_dif2_dat_por[i] = 100*(new_deaths_B2_no_boost[i] - new_deaths_B2_dat[i])/(new_deaths_B2_no_boost[i])

    acum_dif2_mod_por_IC95_1[i] = 100*(new_cases_NB_B2_CI95_1[i] - new_cases_B2_mod[i])/(new_cases_NB_B2_CI95_1[i])
    acum_dif2_dat_por_IC95_1[i] = 100*(new_cases_NB_B2_CI95_1[i] - new_cases_B2_dat[i])/(new_cases_NB_B2_CI95_1[i])
    acum_dif2_mod_por_IC95_2[i] = 100*(new_cases_NB_B2_CI95_2[i] - new_cases_B2_mod[i])/(new_cases_NB_B2_CI95_2[i])
    acum_dif2_dat_por_IC95_2[i] = 100*(new_cases_NB_B2_CI95_2[i] - new_cases_B2_dat[i])/(new_cases_NB_B2_CI95_2[i])

    decea_dif2_mod_por_IC95_1[i] = 100*(new_deaths_NB_B2_CI95_1[i] - new_deaths_B2_mod[i])/(new_deaths_NB_B2_CI95_1[i])
    decea_dif2_dat_por_IC95_1[i] = 100*(new_deaths_NB_B2_CI95_1[i] - new_deaths_B2_dat[i])/(new_deaths_NB_B2_CI95_1[i])
    decea_dif2_mod_por_IC95_2[i] = 100*(new_deaths_NB_B2_CI95_2[i] - new_deaths_B2_mod[i])/(new_deaths_NB_B2_CI95_2[i])
    decea_dif2_dat_por_IC95_2[i] = 100*(new_deaths_NB_B2_CI95_2[i] - new_deaths_B2_dat[i])/(new_deaths_NB_B2_CI95_2[i])

# The data will be saved in .csv files

# Period 1 results
results_B1 = pd.DataFrame()
results_B1["date"] = dates1
results_B1["DifAcumModB1"] = acum_dif1_mod
results_B1["DifAcumDatB1"] = acum_dif1_dat
results_B1["DifFalleModB1"] = decea_dif1_mod
results_B1["DifFalleDatB1"] = decea_dif1_dat
results_B1["DifAcumModB1Por"] = acum_dif1_mod_por
results_B1["DifAcumDatB1Por"] = acum_dif1_dat_por
results_B1["DifFalleModB1Por"] = decea_dif1_mod_por
results_B1["DifFalleDatB1Por"] = decea_dif1_dat_por

# Period 2 results
results_B2 = pd.DataFrame()
results_B2["date"] = dates2
results_B2["DifAcumModB2"] = acum_dif2_mod
results_B2["DifAcumDatB2"] = acum_dif2_dat
results_B2["DifFalleModB2"] = decea_dif2_mod
results_B2["DifFalleDatB2"] = decea_dif2_dat
results_B2["DifAcumModB2Por"] = acum_dif2_mod_por
results_B2["DifAcumDatB2Por"] = acum_dif2_dat_por
results_B2["DifFalleModB2Por"] = decea_dif2_mod_por
results_B2["DifFalleDatB2Por"] = decea_dif2_dat_por

# Total results
results_total = pd.DataFrame()
results_total["date"] = np.concatenate((dates1, dates2))
results_total["DifAcumModTot"] = np.concatenate((acum_dif1_mod, acum_dif2_mod))
results_total["DifAcumDatTot"] = np.concatenate((acum_dif1_dat, acum_dif2_dat))
results_total["DifFalleModTot"] = np.concatenate((decea_dif1_mod, decea_dif2_mod))
results_total["DifFalleDatTot"] = np.concatenate((decea_dif1_dat, decea_dif2_dat))
results_total["DifAcumModTotPor"] = np.concatenate((acum_dif1_mod_por, acum_dif2_mod_por))
results_total["DifAcumDatTotPor"] = np.concatenate((acum_dif1_dat_por, acum_dif2_dat_por))
results_total["DifFalleModTotPor"] = np.concatenate((decea_dif1_mod_por, decea_dif2_mod_por))
results_total["DifFalleDatTotPor"] = np.concatenate((decea_dif1_dat_por, decea_dif2_dat_por))

# Period 1 metrics
metrics_B1 = pd.DataFrame()
metrics_B1["DifFinAcumModB1"] = [acum_dif1_mod[-1]]
metrics_B1["DifFinAcumDatB1"] = [acum_dif1_dat[-1]]
metrics_B1["PromAcumModB1"]   = [np.mean(acum_dif1_mod)]
metrics_B1["PromAcumDatB1"]   = [np.mean(acum_dif1_dat)]
metrics_B1["DifFinFalleModB1"] = [decea_dif1_mod[-1]]
metrics_B1["DifFinFalleDatB1"] = [decea_dif1_dat[-1]]
metrics_B1["PromFalleModB1"]   = [np.mean(decea_dif1_mod)]
metrics_B1["PromFalleDatB1"]   = [np.mean(decea_dif1_dat)]

metrics_B1["DifFinAcumModB1Por"] = [acum_dif1_mod_por[-1]]
metrics_B1["DifFinAcumDatB1Por"] = [acum_dif1_dat_por[-1]]
metrics_B1["PromAcumModB1Por"]   = [np.mean(acum_dif1_mod_por[1:-1])]
metrics_B1["PromAcumDatB1Por"]   = [np.mean(acum_dif1_dat_por[1:-1])]
metrics_B1["DifFinFalleModB1Por"] = [decea_dif1_mod_por[-1]]
metrics_B1["DifFinFalleDatB1Por"] = [decea_dif1_dat_por[-1]]
metrics_B1["PromFalleModB1Por"]   = [np.mean(decea_dif1_mod_por[1:-1])]
metrics_B1["PromFalleDatB1Por"]   = [np.mean(decea_dif1_dat_por[1:-1])]

# Period 2 metrics
metrics_B2 = pd.DataFrame()
metrics_B2["DifFinAcumModB2"] = [(z_NB_b2[:,6][-1] - z_b2[:,6][-1])]
metrics_B2["DifFinAcumDatB2"] = [(z_NB_b2[:,6][-1] - data_tr_b2['Cases'][-1])]
metrics_B2["DifAcumB2DatCI95_1"]  = [z_NB_conf1_b2[:,6][-1] - data_tr_b2['Cases'][-1]]
metrics_B2["DifAcumB2DatCI95_2"]  = [z_NB_conf2_b2[:,6][-1] - data_tr_b2['Cases'][-1]]
metrics_B2["DifAcumB2ModCI95_1"]  = [z_NB_conf1_b2[:,6][-1] - z_b2[:,6][-1]]
metrics_B2["DifAcumB2ModCI95_2"]  = [z_NB_conf2_b2[:,6][-1] - z_b2[:,6][-1]]

metrics_B2["DifFalleB2DatCI95_1"]  = [(z_NB_conf1_b2[:,13][-1] - data_tr_b2['Deaths'][-1])]
metrics_B2["DifFalleB2DatCI95_2"]  = [(z_NB_conf2_b2[:,13][-1] - data_tr_b2['Deaths'][-1])]
metrics_B2["DifFalleB2ModCI95_1"]  = [(z_NB_conf1_b2[:,13][-1] - z_b2[:,13][-1])]
metrics_B2["DifFalleB2ModCI95_2"]  = [(z_NB_conf2_b2[:,13][-1] - z_b2[:,13][-1])]
metrics_B2["PromAcumModB2"]   = [np.mean(new_cases_B2_no_boost - new_cases_B2_mod)]
metrics_B2["PromAcumDatB2"]   = [np.mean(new_cases_B2_no_boost - new_cases_B2_dat)]
metrics_B2["DifFinFalleModB2"] = [(z_NB_b2[:,13][-1] - z_b2[:,13][-1])]
metrics_B2["DifFinFalleDatB2"] = [(z_NB_b2[:,13][-1] - data_tr_b2['Deaths'][-1])]
metrics_B2["PromFalleModB2"]   = [np.mean(new_deaths_B2_no_boost - new_deaths_B2_mod)]
metrics_B2["PromFalleDatB2"]   = [np.mean(new_deaths_B2_no_boost - new_deaths_B2_dat)]

metrics_B2["PromAcumModB2CI95_1"]   = [np.mean(new_cases_NB_B2_CI95_1 - new_cases_B2_mod)]
metrics_B2["PromAcumModB2CI95_2"]   = [np.mean(new_cases_NB_B2_CI95_2 - new_cases_B2_mod)]
metrics_B2["PromAcumDatB2CI95_1"]   = [np.mean(new_cases_NB_B2_CI95_1 - new_cases_B2_dat)]
metrics_B2["PromAcumDatB2CI95_2"]   = [np.mean(new_cases_NB_B2_CI95_2 - new_cases_B2_dat)]
metrics_B2["PromAcumModB2CI95_1Por"]   = [np.mean(acum_dif2_mod_por_IC95_1)]
metrics_B2["PromAcumModB2CI95_2Por"]   = [np.mean(acum_dif2_mod_por_IC95_2)]
metrics_B2["PromAcumDatB2CI95_1Por"]   = [np.mean(acum_dif2_dat_por_IC95_1)]
metrics_B2["PromAcumDatB2CI95_2Por"]   = [np.mean(acum_dif2_dat_por_IC95_2)]

metrics_B2["PromFalleModB2CI95_1"]   = [np.mean(new_deaths_NB_B2_CI95_1 - new_deaths_B2_mod)]
metrics_B2["PromFalleModB2CI95_2"]   = [np.mean(new_deaths_NB_B2_CI95_2 - new_deaths_B2_mod)]
metrics_B2["PromFalleDatB2CI95_1"]   = [np.mean(new_deaths_NB_B2_CI95_1 - new_deaths_B2_dat)]
metrics_B2["PromFalleDatB2CI95_2"]   = [np.mean(new_deaths_NB_B2_CI95_2 - new_deaths_B2_dat)]
metrics_B2["PromFalleModB2CI95_1Por"]   = [np.mean(decea_dif2_mod_por_IC95_1)]
metrics_B2["PromFalleModB2CI95_2Por"]   = [np.mean(decea_dif2_mod_por_IC95_2)]
metrics_B2["PromFalleDatB2CI95_1Por"]   = [np.mean(decea_dif2_dat_por_IC95_1)]
metrics_B2["PromFalleDatB2CI95_2Por"]   = [np.mean(decea_dif2_dat_por_IC95_2)]

metrics_B2["DifFinAcumModB2Por"] = [100*(z_NB_b2[:,6][-1] - z_NB_b2[:,6][0])/(z_b2[:,6][-1] - z_b2[:,6][0])]
metrics_B2["DifFinAcumDatB2Por"] = [100*(z_NB_b2[:,6][-1] - z_NB_b2[:,6][0])/(data_tr_b2['Cases'][-1] - data_tr_b2['Cases'][0])]
metrics_B2["DifAcumB2DatCI95_1Por"]  = [100*(z_NB_conf1_b2[:,6][-1] - z_NB_conf1_b2[:,6][0])/(data_tr_b2['Cases'][-1] - data_tr_b2['Cases'][0] )]
metrics_B2["DifAcumB2DatCI95_2Por"]  = [100*(z_NB_conf2_b2[:,6][-1] - z_NB_conf2_b2[:,6][0])/(data_tr_b2['Cases'][-1] - data_tr_b2['Cases'][0])]
metrics_B2["DifAcumB2ModCI95_1Por"]  = [100*(z_NB_conf1_b2[:,6][-1] - z_NB_conf1_b2[:,6][0])/(z_b2[:,6][-1] - z_b2[:,6][0] )]
metrics_B2["DifAcumB2ModCI95_2Por"]  = [100*(z_NB_conf2_b2[:,6][-1] - z_NB_conf2_b2[:,6][0])/(z_b2[:,6][-1] - z_b2[:,6][0] )]
metrics_B2["PromAcumModB2Por"]   = [np.mean(acum_dif2_mod_por[1:])]
metrics_B2["PromAcumDatB2Por"]   = [np.mean(acum_dif2_dat_por[1:])]

metrics_B2["DifFinFalleModB2Por"] = [100*(z_NB_b2[:,13][-1] - z_NB_b2[:,13][0])/(z_b2[:,13][-1] - z_b2[:,13][0])]
metrics_B2["DifFinFalleDatB2Por"] = [100*(z_NB_b2[:,13][-1] - z_NB_b2[:,13][0])/(data_tr_b2['Deaths'][-1] - data_tr_b2['Deaths'][0])]
metrics_B2["DifFalleB2DatCI95_1Por"]  = [100*(z_NB_conf1_b2[:,13][-1] - z_NB_conf1_b2[:,13][0])/(data_tr_b2['Deaths'][-1] - data_tr_b2['Deaths'][0] )]
metrics_B2["DifFalleB2DatCI95_2Por"]  = [100*(z_NB_conf2_b2[:,13][-1] - z_NB_conf2_b2[:,13][0])/(data_tr_b2['Deaths'][-1] - data_tr_b2['Deaths'][0] )]
metrics_B2["DifFalleB2ModCI95_1Por"]  = [100*(z_NB_conf1_b2[:,13][-1] - z_NB_conf1_b2[:,13][0])/(z_b2[:,13][-1] - z_b2[:,13][0] )]
metrics_B2["DifFalleB2ModCI95_2Por"]  = [100*(z_NB_conf2_b2[:,13][-1] - z_NB_conf2_b2[:,13][0])/(z_b2[:,13][-1] - z_b2[:,13][0] )]
metrics_B2["PromFalleModB2Por"]   = [np.mean(decea_dif2_mod_por[1:])]
metrics_B2["PromFalleDatB2Por"]   = [np.mean(decea_dif2_dat_por[1:])]


# Total metrics
metrics_total = pd.DataFrame()
metrics_total["DifFinAcumModTot"] = [acum_dif2_mod[-1]]
metrics_total["DifFinAcumDatTot"] = [acum_dif2_dat[-1]]
metrics_total["PromAcumModTot"]   = [np.mean(results_total["DifAcumModTot"])]
metrics_total["PromAcumDatTot"]   = [np.mean(results_total["DifAcumDatTot"])]
metrics_total["DifFinFalleModTot"] = [decea_dif2_mod[-1]]
metrics_total["DifFinFalleDatTot"] = [decea_dif2_dat[-1]]
metrics_total["PromFalleModTot"]   = [np.mean(results_total["DifFalleModTot"])]
metrics_total["PromFalleDatTot"]   = [np.mean(results_total["DifFalleDatTot"])]

metrics_total["DifFinAcumModTotPor"] = [acum_dif2_mod_por[-1]]
metrics_total["DifFinAcumDatTotPor"] = [acum_dif2_dat_por[-1]]
metrics_total["PromAcumModTotPor"]   = [np.mean(results_total["DifAcumModTotPor"])]
metrics_total["PromAcumDatTotPor"]   = [np.mean(results_total["DifAcumDatTotPor"])]
metrics_total["DifFinFalleModTotPor"] = [decea_dif2_mod_por[-1]]
metrics_total["DifFinFalleDatTotPor"] = [decea_dif2_dat_por[-1]]
metrics_total["PromFalleModTotPor"]   = [np.mean(results_total["DifFalleModTotPor"])]
metrics_total["PromFalleDatTotPor"]   = [np.mean(results_total["DifFalleDatTotPor"])]


# Saved DataFrames as .csv
# Uncomment when the destiny paths are correct
# The data will be printed anyway
results_B1.to_csv("Results/resultsB1.csv", index=False)
results_B2.to_csv("Results/resultsB2.csv", index=False)
results_total.to_csv("Results/resultsTotal.csv", index=False)

metrics_B1.to_csv("Results/metricsB1.csv", index=False)
metrics_B2.to_csv("Results/metricsB2.csv", index=False)
metrics_total.to_csv("Results/metricsTotal.csv", index=False)

print("Bloque 1:")
print(f'Diferencia acumulados modelo: {metrics_B1["DifFinAcumModB1"].values[0]}')
print(f'Diferencia acumulados datos : {metrics_B1["DifFinAcumDatB1"].values[0]}')
print(f'Promedio acumulados modelo  : {metrics_B1["PromAcumModB1"].values[0]}')
print(f'Promedio acumulados datos   : {metrics_B1["PromAcumDatB1"].values[0]}')
print(f'Diferencia fallecidos modelo: {metrics_B1["DifFinFalleModB1"].values[0]}')
print(f'Diferencia fallecidos datos : {metrics_B1["DifFinFalleDatB1"].values[0]}')
print(f'Promedio fallecidos modelo  : {metrics_B1["PromFalleModB1"].values[0]}')
print(f'Promedio fallecidos datos   : {metrics_B1["PromFalleDatB1"].values[0]}')

print(f'Diferencia acumulados modelo, porcentual: {metrics_B1["DifFinAcumModB1Por"].values[0]}')
print(f'Diferencia acumulados datos, porcentual : {metrics_B1["DifFinAcumDatB1Por"].values[0]}')
print(f'Promedio acumulados modelo, porcentual  : {metrics_B1["PromAcumModB1Por"].values[0]}')
print(f'Promedio acumulados datos, porcentual   : {metrics_B1["PromAcumDatB1Por"].values[0]}')
print(f'Diferencia fallecidos modelo, porcentual: {metrics_B1["DifFinFalleModB1Por"].values[0]}')
print(f'Diferencia fallecidos datos, porcentual : {metrics_B1["DifFinFalleDatB1Por"].values[0]}')
print(f'Promedio fallecidos modelo, porcentual  : {metrics_B1["PromFalleModB1Por"].values[0]}')
print(f'Promedio fallecidos datos, porcentual   : {metrics_B1["PromFalleDatB1Por"].values[0]}')

print()

print("Bloque 2:")
print(f'Diferencia acumulados modelo: {metrics_B2["DifFinAcumModB2"].values[0]}')
print(f'Diferencia acumulados datos : {metrics_B2["DifFinAcumDatB2"].values[0]}')
print(f'Promedio acumulados modelo  : {metrics_B2["PromAcumModB2"].values[0]}')
print(f'Promedio acumulados datos   : {metrics_B2["PromAcumDatB2"].values[0]}')

print()

print(f'Diferencia fallecidos modelo: {metrics_B2["DifFinFalleModB2"].values[0]}')
print(f'Diferencia fallecidos datos : {metrics_B2["DifFinFalleDatB2"].values[0]}')
print(f'Promedio fallecidos modelo  : {metrics_B2["PromFalleModB2"].values[0]}')
print(f'Promedio fallecidos datos   : {metrics_B2["PromFalleDatB2"].values[0]}')

print()

print(f'Diferencia acumulados modelo, porcentual: {metrics_B2["DifFinAcumModB2Por"].values[0]}')
print(f'Diferencia acumulados datos, porcentual : {metrics_B2["DifFinAcumDatB2Por"].values[0]}')
print(f'Promedio acumulados modelo, porcentual  : {metrics_B2["PromAcumModB2Por"].values[0]}')
print(f'Promedio acumulados datos, porcentual   : {metrics_B2["PromAcumDatB2Por"].values[0]}')

print()

print(f'Diferencia fallecidos modelo, porcentual: {metrics_B2["DifFinFalleModB2Por"].values[0]}')
print(f'Diferencia fallecidos datos, porcentual : {metrics_B2["DifFinFalleDatB2Por"].values[0]}')
print(f'Promedio fallecidos modelo, porcentual  : {metrics_B2["PromFalleModB2Por"].values[0]}')
print(f'Promedio fallecidos datos, porcentual   : {metrics_B2["PromFalleDatB2Por"].values[0]}')

print()

print(f'Diferencia de acumulados datos, IC95 1:    {metrics_B2["DifAcumB2DatCI95_1"].values[0]}')
print(f'Diferencia de acumulados datos, IC95 2:    {metrics_B2["DifAcumB2DatCI95_2"].values[0]}')
print(f'Diferencia de acumulados modelo, IC95 1:    {metrics_B2["DifAcumB2ModCI95_1"].values[0]}')
print(f'Diferencia de acumulados modelo, IC95 2:    {metrics_B2["DifAcumB2ModCI95_2"].values[0]}')

print()

print(f'Diferencia de acumulados datos, IC95 1, porcentual:    {metrics_B2["DifAcumB2DatCI95_1Por"].values[0]}')
print(f'Diferencia de acumulados datos, IC95 2, porcentual:    {metrics_B2["DifAcumB2DatCI95_2Por"].values[0]}')
print(f'Diferencia de acumulados modelo, IC95 1, porcentual:    {metrics_B2["DifAcumB2ModCI95_1Por"].values[0]}')
print(f'Diferencia de acumulados modelo, IC95 2, porcentual:    {metrics_B2["DifAcumB2ModCI95_2Por"].values[0]}')

print()

print(f'Diferencia de fallecidos datos, IC95 1:    {metrics_B2["DifFalleB2DatCI95_1"].values[0]}')
print(f'Diferencia de fallecidos datos, IC95 2:    {metrics_B2["DifFalleB2DatCI95_2"].values[0]}')
print(f'Diferencia de fallecidos modelo, IC95 1:    {metrics_B2["DifFalleB2ModCI95_1"].values[0]}')
print(f'Diferencia de fallecidos modelo, IC95 2:    {metrics_B2["DifFalleB2ModCI95_2"].values[0]}')

print()

print(f'Diferencia de fallecidos datos, IC95 1, porcentual:    {metrics_B2["DifFalleB2DatCI95_1Por"].values[0]}')
print(f'Diferencia de fallecidos datos, IC95 2, porcentual:    {metrics_B2["DifFalleB2DatCI95_2Por"].values[0]}')
print(f'Diferencia de fallecidos modelo, IC95 1, porcentual:    {metrics_B2["DifFalleB2ModCI95_1Por"].values[0]}')
print(f'Diferencia de fallecidos modelo, IC95 2, porcentual:    {metrics_B2["DifFalleB2ModCI95_2Por"].values[0]}')

print()

print(f'Promedio de casos datos, IC95 1:    {metrics_B2["PromAcumDatB2CI95_1"].values[0]}')
print(f'Promedio de casos datos, IC95 2:    {metrics_B2["PromAcumDatB2CI95_2"].values[0]}')
print(f'Promedio de casos modelo, IC95 1:    {metrics_B2["PromAcumModB2CI95_1"].values[0]}')
print(f'Promedio de casos modelo, IC95 2:    {metrics_B2["PromAcumModB2CI95_2"].values[0]}')

print()

print(f'Promedio de casos datos porcentual IC95 1:    {metrics_B2["PromAcumDatB2CI95_1Por"].values[0]}')
print(f'Promedio de casos datos porcentual IC95 2:    {metrics_B2["PromAcumDatB2CI95_2Por"].values[0]}')
print(f'Promedio de casos modelo porcentual IC95 1:    {metrics_B2["PromAcumModB2CI95_1Por"].values[0]}')
print(f'Promedio de casos modelo porcentual IC95 2:    {metrics_B2["PromAcumModB2CI95_2Por"].values[0]}')

print()

print(f'Promedio de fallecidos datos, IC95 1:    {metrics_B2["PromFalleDatB2CI95_1"].values[0]}')
print(f'Promedio de fallecidos datos, IC95 2:    {metrics_B2["PromFalleDatB2CI95_2"].values[0]}')
print(f'Promedio de fallecidos modelo, IC95 1:    {metrics_B2["PromFalleModB2CI95_1"].values[0]}')
print(f'Promedio de fallecidos modelo, IC95 2:    {metrics_B2["PromFalleModB2CI95_2"].values[0]}')

print()

print(f'Promedio de fallecidos datos porcentual IC95 1:    {metrics_B2["PromFalleDatB2CI95_1Por"].values[0]}')
print(f'Promedio de fallecidos datos porcentual IC95 2:    {metrics_B2["PromFalleDatB2CI95_2Por"].values[0]}')
print(f'Promedio de fallecidos modelo porcentual IC95 1:    {metrics_B2["PromFalleModB2CI95_1Por"].values[0]}')
print(f'Promedio de fallecidos modelo porcentual IC95 2:    {metrics_B2["PromFalleModB2CI95_2Por"].values[0]}')

# Estimated total population in boot vaccine scenario
SN    = z_b2[-1,0]
I_aN  = z_b2[-1,1]
I_sN  = z_b2[-1,2]
R_aN  = z_b2[-1,3]
R_sN  = z_b2[-1,4]
DN    = z_b2[-1,5]
Ac    = z_b2[-1,6]
SV    = z_b2[-1,7]
I_aV  = z_b2[-1,8]
I_sV  = z_b2[-1,9]
R_aV  = z_b2[-1,10]
R_sV  = z_b2[-1,11]
DV    = z_b2[-1,12]
D     = z_b2[-1,13]
SVr   = z_b2[-1,14]
I_aVr = z_b2[-1,15]
I_sVr = z_b2[-1,16]
R_aVr = z_b2[-1,17]
R_sVr = z_b2[-1,18]
DVr   = z_b2[-1,19]

# Total population
NT  = (SN + I_aN + I_sN + R_aN + R_sN) +(SV + I_aV + I_sV + R_aV + R_sV) +(SVr + I_aVr + I_sVr + R_aVr + R_sVr);

print()
print(f"Población total: {NT}")

print(f"Porcentaje población en casos, modelo: {100*metrics_B2['DifFinAcumModB2'].values[0]/NT}")
print(f"Porcentaje población en casos, datos: {100*metrics_B2['DifFinAcumDatB2'].values[0]/NT}")
print(f"Porcentaje población en muertes, modelo: {100*metrics_B2['DifFinFalleModB2'].values[0]/NT}")
print(f"Porcentaje población en muertes, datos: {100*metrics_B2['DifFinFalleDatB2'].values[0]/NT}")

# Plots

# Plots for the first period

min0 = min(z_b1[:,6][0], z_NB_b1[:,6][0], data_tr_b1['Cases'][0])

# Accumulated cases standard
plt.figure(figsize=(16,9))
plt.plot(dates1, z_b1[:,6] - min0      ,'r' , label='Boost (model)')
plt.plot(dates1, z_NB_b1[:,6] - min0    ,'b' , label='No boost (model)')
plt.plot(dates1, data_tr_b1['Cases'] - min0,'.k', label='Data')
plt.title('Cumulated cases between ' + t0_b1 + " and " + tf_b1 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/AcumB1From0.png")
plt.show()

min1 = min(z_b1[:,6][0], z_NB_b1[:,6][0], data_tr_b1['Cases'][0], z_conf1_b1[:,6][0],  z_conf2_b1[:,6][0])

# Accumulated cases standard
plt.figure(figsize=(16,9))
plt.plot(dates1, z_b1[:,6] - min1      ,'r' , label='Boost (model)')
plt.plot(dates1, z_NB_b1[:,6] - min1    ,'b' , label='No Boost (model)')
plt.plot(dates1, data_tr_b1['Cases'] - min1,'.k', label='Data')
plt.plot(dates1, z_NB_conf1_b1[:,6] - min1,'orange' , label='CI95 (No boost)')
plt.plot(dates1, z_NB_conf2_b1[:,6] - min1,'orange')
plt.fill_between(dates1, z_NB_conf1_b1[:,6] - min1, z_NB_conf2_b1[:,6] -min1, color="orange", alpha=0.4)
plt.title('Cumulated cases between ' + t0_b1 + " and " + tf_b1 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/AcumB1ConfIntFrom0.png")
plt.show()

# Daily cases, with confidence interval
plt.figure(figsize=(16,9))
plt.plot(dates1[1:], new_cases_B1_mod          ,'r' , label='Boost (model)')
plt.plot(dates1[1:], new_cases_B1_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates1[1:], new_cases_B1_dat,'k', label='Data')
plt.title('Daily cases between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/NewCasesB1.png")
plt.show()

# Daily cases, with confidence interval
plt.figure(figsize=(16,9))
plt.plot(dates1[1:], new_cases_B1_mod          ,'r' , label='Boost (model)')
plt.plot(dates1[1:], new_cases_B1_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates1[1:], new_cases_B1_dat,'k', label='Data')
plt.plot(dates1[1:], new_cases_NB_B1_CI95_1 ,'orange' , label='IC95 (No boost)')
plt.plot(dates1[1:], new_cases_NB_B1_CI95_2,'orange')
plt.fill_between(dates1[1:], new_cases_B1_CI95_1, new_cases_B1_CI95_2, color='orange', alpha=0.4)
plt.title('Daily cases between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/NewCasesConfIntB1.png")
plt.show()

# Daily deaths, with confidence interval
plt.figure(figsize=(16,9))
plt.plot(dates1[1:], new_deaths_B1_mod          ,'r' , label='Boost (model)')
plt.plot(dates1[1:], new_deaths_B1_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates1[1:], new_deaths_B1_dat,'k', label='Data')
plt.title('Daily deaths between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily deaths")
plt.legend()
plt.tight_layout()
# # Uncomment if want to save the figure
# # Beware with the destiny path and the file's name
plt.savefig("Results/NewDeathsB1.png")
plt.show()

# Daily deaths, with confidence interval
plt.figure(figsize=(16,9))
plt.plot(dates1[1:], new_deaths_B1_mod          ,'r' , label='Boost (model)')
plt.plot(dates1[1:], new_deaths_B1_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates1[1:], new_deaths_B1_dat,'k', label='Data')
plt.plot(dates1[1:], new_deaths_NB_B1_CI95_1 ,'orange' , label='CI95 (No boost)')
plt.plot(dates1[1:], new_deaths_NB_B1_CI95_2,'orange')
plt.fill_between(dates1[1:], new_deaths_B1_CI95_1, new_deaths_B1_CI95_2, color='orange', alpha=0.4)
plt.title('Daily deaths between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily deaths")
plt.legend()
plt.tight_layout()
# # Uncomment if want to save the figure
# # Beware with the destiny path and the file's name
plt.savefig("Results/NewDeathsB1ConfInt.png")
plt.show()



# Deceased
min2 = min(z_b1[:,13][0], z_NB_b1[:,13][0], data_tr_b1['Deaths'][0], z_NB_conf1_b1[:,13][0],  z_NB_conf2_b1[:,13][0])

plt.figure(figsize=(16,9))
plt.plot(dates1, z_b1[:,13] - min2      ,'r' , label='Boost (model)')
plt.plot(dates1, z_NB_b1[:,13] - min2    ,'b' , label='No boost (model)')
plt.plot(dates1, data_tr_b1['Deaths'] - min2,'.k', label='Data')
plt.title('Cumulated deaths between ' + t0_b1 + " and " + tf_b1 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated deaths")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/FalleB1From0.png")
plt.show()

# Deceased
plt.figure(figsize=(16,9))
plt.plot(dates1, z_b1[:,13] - min2      ,'r' , label='Boost (model)')
plt.plot(dates1, z_NB_b1[:,13] - min2    ,'b' , label='No boost (model)')
plt.plot(dates1, data_tr_b1['Deaths'] - min2,'.k', label='Data')
plt.plot(dates1, z_NB_conf1_b1[:,13] - min2, "orange", label="CI95 (No boost)")
plt.plot(dates1, z_NB_conf2_b1[:,13] - min2, "orange")
plt.fill_between(dates1, z_NB_conf1_b1[:,13] - min2, z_NB_conf2_b1[:,13] - min2, color="orange", alpha=0.4)
plt.title('Cumulated deaths between ' + t0_b1 + " and " + tf_b1 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated deaths")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/FalleB1ConfIntFrom0.png")
plt.show()

# Plots for the second period

min3 = min(z_b2[:,6][0], z_NB_b2[:,6][0], data_tr_b2['Cases'][0], z_NB_conf1_b2[:,6][0],  z_NB_conf2_b2[:,6][0])

# Accumulated cases, standard
plt.figure(figsize=(16,9))
plt.plot(dates2, z_b2[:,6]  - min3    ,'r' , label='Boost(model)')
plt.plot(dates2, z_NB_b2[:,6] - min3    ,'b' , label='No boost (model)')
plt.plot(dates2, data_tr_b2['Cases'] - min3,'.k', label='Data')
plt.title('Cumulated cases between ' + t0_b2 + " and " + tf_b2 + " starting from 0")
plt.xlabel("Date", fontsize=14)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/AcumB2From0.png")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(dates2, z_b2[:,6]  - min3    ,'r' , label='Boost(model)')
plt.plot(dates2, z_NB_b2[:,6] - min3    ,'b' , label='No boost (model)')
plt.plot(dates2, data_tr_b2['Cases'] - min3,'.k', label='Data')
plt.plot(dates2, z_NB_conf1_b2[:,6] - min3, 'orange', label="IC95 (No boost)" )
plt.plot(dates2, z_NB_conf2_b2[:,6] - min3, 'orange')
plt.fill_between(dates2, z_NB_conf1_b2[:,6] - min3, z_NB_conf2_b2[:,6] - min3, color='orange', alpha=0.4)
plt.title('Cumulated cases between ' + t0_b2 + " and " + tf_b2 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/AcumB2ConfIntFrom0.png")
plt.show()


# Daily cases
plt.figure(figsize=(16,9))
plt.plot(dates2[1:], new_cases_B2_mod          ,'r' , label='Boost (model)')
plt.plot(dates2[1:], new_cases_B2_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates2[1:], new_cases_B2_dat,'k', label='Data')
plt.title('Daily cases between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/NewCasesB2.png")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(dates2[1:], new_cases_B2_mod          ,'r' , label='Boost (model)')
plt.plot(dates2[1:], new_cases_B2_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates2[1:], new_cases_B2_dat,'k', label='Data')
plt.plot(dates2[1:], new_cases_NB_B2_CI95_1 ,'orange' , label='IC95 (No boost)')
plt.plot(dates2[1:], new_cases_NB_B2_CI95_2,'orange')
plt.fill_between(dates2[1:], new_cases_NB_B2_CI95_1, new_cases_NB_B2_CI95_2, color='orange', alpha=0.4)
plt.title('Daily cases between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily cases")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/NewCasesB2ConfInt.png")
plt.show()

# Daily deaths
plt.figure(figsize=(16,9))
plt.plot(dates2[1:], new_deaths_B2_mod          ,'r' , label='Boost (model)')
plt.plot(dates2[1:], new_deaths_B2_no_boost     ,'b' , label='No boost (model)')
plt.plot(dates2[1:], new_deaths_B2_dat,'k', label='Data')
plt.plot(dates2[1:], new_deaths_NB_B2_CI95_1 ,'orange' , label='IC95: Con refuerzo')
plt.plot(dates2[1:], new_deaths_NB_B2_CI95_2,'orange')
plt.fill_between(dates2[1:], new_deaths_NB_B2_CI95_1, new_deaths_NB_B2_CI95_2, color="orange", alpha=0.4)
plt.title('Daily deaths between ' + t0_b1 + " and " + tf_b1)
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Daily deaths")
plt.legend()
plt.tight_layout()
# Uncomment if want to save the figure
# Beware with the destiny path and the file's name
plt.savefig("Results/NewDeathsConfIntB2.png")
plt.show()


min4 = min(z_b2[:,13][0], z_NB_b2[:,13][0], data_tr_b2['Deaths'][0], z_NB_conf1_b2[:,13][0],  z_NB_conf2_b2[:,13][0])

plt.figure(figsize=(16,9))
plt.plot(dates2, z_b2[:,13] - min4     ,'r' , label='Boost (model)')
plt.plot(dates2, z_NB_b2[:,13]  - min4   ,'b' , label='No boost (model)')
plt.plot(dates2, data_tr_b2['Deaths'] - min4,'.k', label='Data')
plt.title('Cumulated deaths between ' + t0_b2 + " and " + tf_b2 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated deaths")
plt.legend()
plt.tight_layout()
# # Uncomment if want to save the figure
# # Beware with the destiny path and the file's name
plt.savefig("Results/FalleB2From0.png")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(dates2, z_b2[:,13] - min4     ,'r' , label='Boost (model)')
plt.plot(dates2, z_NB_b2[:,13]  - min4   ,'b' , label='No boost (model)')
plt.plot(dates2, data_tr_b2['Deaths'] - min4,'.k', label='Data')
plt.plot(dates2, z_NB_conf1_b2[:,13] - min4, 'orange', label="IC95: Sin refuerzo" )
plt.plot(dates2, z_NB_conf2_b2[:,13] - min4, 'orange')
plt.fill_between(dates2, z_NB_conf1_b2[:,13] - min4, z_NB_conf2_b2[:,13] - min4, color='orange', alpha=0.4)
plt.title('Cumulated deaths between ' + t0_b2 + " and " + tf_b2 + " starting from 0")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
plt.ylabel("Cumulated deaths")
plt.legend()
plt.tight_layout()
# # Uncomment if want to save the figure
# # Beware with the destiny path and the file's name
plt.savefig("Results/FalleB2ConfIntFrom0.png")
plt.show()