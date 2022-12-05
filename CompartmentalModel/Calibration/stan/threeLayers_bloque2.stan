functions {

    real linear_interpolation(real x, real[] x_pred, real[] y_pred){
        int K = size(x_pred);
        real deltas [K];
        real ans;
        int i;
        real t;
        real w;
        real x1;
        real x2;
        real y1;
        real y2;
        if(size(y_pred) != K) reject("x_pred and y_pred aren't of the same size");
        for(k in 1:K) deltas[k] = x - x_pred[k];
        i =sort_indices_asc(fabs(deltas))[1];
        if(deltas[i] < 0|| i==K)i-=1;
        x1 = x_pred[i];
        x2 = x_pred[i + 1];
        y1 = y_pred[i];
        y2 = y_pred[i + 1];
        t = (x-x1)/(x2-x1);
        w = 1-t;
        //w = 1/(1 + exp(1/(1-t) - 1/t));
        //w = 1 - 3*pow(t,2) + 2*pow(t,3);
        ans = w*y1 + (1-w)*y2;

    return(ans);
  }

    real[] dz_dt(real t, real[] z, real[] theta,real[] x_r, int[] x_i){

        // parameters of the compartmental model
        // first layer
        real beta_aN = theta[1];// rate of transmission for asintomatic infection  NO vaccinated
        real beta_sN = theta[2];// rate of transmission for sintomatic infection  NO vaccinated

        // second layer

        real beta_aV = theta[3];//percentage of beta_aN
        real beta_sV = theta[4];//percentage of beta_sN

        // third layer
        real gamma_aVr= theta[5];// recover rate for asintomatic people vaccinated
        real gamma_sVr= theta[6];// recover rate for sintomatic people vaccinated
        real deltaVr  = theta[7];// infection fatality rate of people vaccinated (only for sintomatic people)
        real beta_aVr = theta[8];//percentage of beta_aN
        real beta_sVr = theta[9];//percentage of beta_sN
        real subr = theta[10];//subreporte


        //fixed
        int N_pred   = x_i[1];
        real pN      = x_r[5*N_pred+1];// probability of being asintomatic when NO vaccinated
        real q_aN    = x_r[5*N_pred+2];// rate of reinfection for asintomatic infection NO vaccinated
        real q_sN    = x_r[5*N_pred+3];// rate of reinfection for sintomatic infection  NO vaccinated
        real pV      = x_r[5*N_pred+4];// probability of being asintomatic when vaccinate
        real q_aV    = x_r[5*N_pred+5];// rate of reinfection for asintomatic infectionof vaccinated
        real q_sV    = x_r[5*N_pred+6];// rate of reinfection for sintomatic infection of vaccinated

        real pVr      = x_r[5*N_pred+7];// probability of being asintomatic when vaccinate booster
        real q_aVr    = x_r[5*N_pred+8];// rate of reinfection for asintomatic infectionof vaccinated booster
        real q_sVr    = x_r[5*N_pred+9];// rate of reinfection for sintomatic infection of vaccinated booster


        real ut      = x_r[5*N_pred+10];// proportion of asintomatic people isolated before transmission of the disease
        real mu      = x_r[5*N_pred+11];// proportion of sintomatic people isolated before transmission of the disease
        real gamma_aN= x_r[5*N_pred+12];// recover rate for asintomatic people NO vaccinated
        real gamma_sN= x_r[5*N_pred+13];// recover rate for sintomatic people NO vaccinated
        real deltaN  = x_r[5*N_pred+14];// infection fatality rate (only for sintomatic people) for NO vaccinated
        real gamma_aV= x_r[5*N_pred+15];// recover rate for asintomatic people vaccinated
        real gamma_sV= x_r[5*N_pred+16];// recover rate for sintomatic people vaccinated
        real deltaV  = x_r[5*N_pred+17];// infection fatality rate of people vaccinated (only for sintomatic people)
        real S0V_an =x_r[5*N_pred+18];
        real R_a0V_an =x_r[5*N_pred+19];
        real R_s0V_an =x_r[5*N_pred+20];


        // values of the states of the compartments
        // first layer:  NO vaccinated people
        real SN   = z[1]; // number of people susceptible
        real I_aN = z[2]; // number of people infectious and asintomatics
        real I_sN = z[3]; // number of people infectious and sintomatics
        real R_aN = z[4]; // number of people recovered and from asintomatic infection
        real R_sN = z[5]; // number of people recovered and from sintomatic infection
        real DN   = z[6]; // number of deaths of No vaccinated

        real Ac  = z[7];  // cumulative sintomatics cases

        // second layer: vaccinated people
        real SV   = z[8]; // number of people susceptible
        real I_aV = z[9]; // number of people infectious and asintomatics
        real I_sV = z[10]; // number of people infectious and sintomatics
        real R_aV = z[11]; // number of people recovered and from asintomatic infection
        real R_sV = z[12]; // number of people recovered and from sintomatic infection
        real DV   = z[13]; // number of deaths of No vaccinated

        real D    = z[14];  // cumulative deaths

        //third layer: booster
        real SVr   = z[15]; // number of people susceptible
        real I_aVr = z[16]; // number of people infectious and asintomatics
        real I_sVr = z[17]; // number of people infectious and sintomatics
        real R_aVr = z[18]; // number of people recovered and from asintomatic infection
        real R_sVr = z[19]; // number of people recovered and from sintomatic infection
        real DVr   = z[20]; // number of deaths of No vaccinated


        //function f_V(t)
        real ft =linear_interpolation(t,x_r[1:N_pred],x_r[(N_pred+1):2*N_pred]);//factor de la efectividad acumulada

        real ft2 =linear_interpolation(t,x_r[1:N_pred],x_r[(2*N_pred+1):3*N_pred]);//factor de la efectividad acumulada
        real v   =linear_interpolation(t,x_r[1:N_pred],x_r[3*N_pred+1:4*N_pred]); // velocity of vaccinated with 2 shots o unique shot each day

        real vr =linear_interpolation(t,x_r[1:N_pred],x_r[4*N_pred+1:5*N_pred]); // velocity of vaccinated booster


       // real vr=66242.24193548386;

        // number of people alive
        real NT  = (SN + I_aN + I_sN + R_aN + R_sN) + (SV + I_aV + I_sV + R_aV + R_sV)+(SVr + I_aVr + I_sVr + R_aVr + R_sVr);

        // new No vaccinated infected
        real LambdaN = ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SN / NT;

        // new vaccinated infected
        real LambdaV = (1-ft) *  ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SV / NT;

        // new booster infected
        real LambdaVr = (1-ft2) * ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV+ beta_aVr * beta_aV * beta_aN * I_aVr) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV+ beta_sVr * beta_sV * beta_sN * I_sVr)) * SVr / NT;


        // new vaccinated in the compartment: susceptible vaccinated
        real vSVN = v * SN / (SN + R_aN + R_sN);
        // new vaccinated in the compartment: recovered asintomatico
        real vRNa = v * R_aN / (SN + R_aN + R_sN);
        // new vaccinated in the compartment : recovered sintomatico
        real vRNs = v * R_sN / (SN + R_aN + R_sN);

    //booster
        // new vaccinated in the compartment: susceptible vaccinated
        //real vSVNr = vr * SV / (SV + R_aV + R_sV);
        real vSVNr = vr * S0V_an / (S0V_an + R_a0V_an + R_s0V_an);
        //real vSVNr = vr*0.967;

        // new vaccinated in the compartment: recovered asintomatico
        // real vRNar = vr * R_aV / (SV + R_aV + R_sV);
        real vRNar = vr * R_a0V_an/ (S0V_an + R_a0V_an + R_s0V_an);
        //real vRNar = vr*0.02;

        // new vaccinated in the compartment : recovered sintomatico
        //real vRNsr = vr * R_sV / (SV + R_aV + R_sV);
        real vRNsr = vr * R_s0V_an / (S0V_an + R_a0V_an + R_s0V_an);
        //real vRNsr = vr * 0.013;

        return {
        //update first layer
            -LambdaN+q_aN*R_aN+q_sN*R_sN-vSVN,
            pN*LambdaN-gamma_aN*I_aN,
            (1-pN)*LambdaN-(gamma_sN+deltaN)*I_sN,
            gamma_aN*I_aN-q_aN*R_aN-vRNa,
            gamma_sN*I_sN-q_sN*R_sN-vRNs,
            deltaN*I_sN,

            // cumulative sintomatics cases
            subr*((1-pN)*LambdaN+(1-pV)*LambdaV+(1-pVr)*LambdaVr),

            // update second layer
            -LambdaV+q_aV*R_aV+q_sV*R_sV+v-vSVNr,
            pV*LambdaV-gamma_aV*I_aV,
            (1-pV)*LambdaV-(gamma_sV+deltaV)*I_sV,
            gamma_aV*I_aV-q_aV*R_aV-vRNar,
            gamma_sV*I_sV-q_sV*R_sV-vRNsr,
            deltaV*I_sV,

            deltaN*I_sN+deltaV*I_sV+deltaVr*I_sVr, // cumulative deaths

            // update third layer
            -LambdaVr+q_aVr*R_aVr+q_sVr*R_sVr+vr,
            pVr*LambdaVr-gamma_aVr*I_aVr,
            (1-pVr)*LambdaVr-(gamma_sVr+deltaVr)*I_sVr,
            gamma_aVr*I_aVr-q_aVr*R_aVr,
            gamma_sVr*I_sVr-q_sVr*R_sVr,
            deltaVr*I_sVr
            };
        }
    }

data {
    int<lower=0> N; // number of days
    real T[N]; //time steps
    int<lower=0> N_pred;
    real T_pred[N_pred];

    // observed data
    real yIs[N]; // cumulative number of sintomaticos infectious peopole
    real yD[N]; // deaths
    real yV[N_pred]; // vaccinated se cambió acá
    real yVr[N_pred]; // vaccinated se cambió acá
    real ye[N_pred]; // ft function from efectividad.py
    real yer[N_pred]; // ft function from efectividad.py


    // parameters we have set
    real<lower=0> N0; // population in Chile
    real<lower=0> mu; // proportion of detected cases that are isolated in time and that will not transmit
    real<lower=0> ut; // proportion of asintomatic cases detected and isolated in time and that will not transmit
    real<lower=0> q_aN; // rate of reinfection for asintomatic infection NO vaccinated
    real<lower=0> q_sN; // rate of reinfection for sintomatic infection NO vaccinated
    real<lower=0> pN; // population no vaccinated in Chile
    real<lower=0> q_aV; // rate of reinfection for asintomatic infection of vaccinated
    real<lower=0> q_sV; // rate of reinfection for sintomatic infection of vaccinated
    real<lower=0> pV; // probability of being asintomatic when vaccinated
    real<lower=0> q_aVr; // rate of reinfection for asintomatic infection of vaccinated
    real<lower=0> q_sVr; // rate of reinfection for sintomatic infection of vaccinated
    real<lower=0> pVr; // probability of being asintomatic when vaccinated

    real<lower=0> S0N_anterior; // S0 final del mes anterior no vacunados
    real<lower=0> D0N_anterior;
    real<lower=0> gamma_aN;
    real<lower=0> gamma_sN;
    real<lower=0> deltaN;
    real<lower=0> I_a0N_anterior;
    real<lower=0> I_s0N_anterior;
    real<lower=0> R_a0N_anterior;
    real<lower=0> R_s0N_anterior;

    real<lower=0> S0V_anterior; // S0 final del mes anterior no vacunados
    real<lower=0> gamma_aV;
    real<lower=0> gamma_sV;
    real<lower=0> deltaV;
    real<lower=0> I_a0V_anterior;
    real<lower=0> I_s0V_anterior;
    real<lower=0> R_a0V_anterior;
    real<lower=0> R_s0V_anterior;
    real<lower=0> D0V_anterior;
    real<lower=0> S0Vr_anterior;
    real<lower=0> I_a0Vr_anterior;
    real<lower=0> I_s0Vr_anterior;
    real<lower=0> R_a0Vr_anterior;
    real<lower=0> R_s0Vr_anterior;
    real<lower=0> D0Vr_anterior;
}

transformed data { // we use it to solve the ODE
    real x_r[5*N_pred+20];
    int x_i[1];
    real rel_tol = 1e-6;
    real abs_tol = 1e-6;
    real max_num_steps = 1e4;
    x_i[1]=N_pred;
    x_r[1:N_pred]=T_pred;
    x_r[N_pred+1:2*N_pred]=ye;
    x_r[(2*N_pred)+1:3*N_pred]=yer;
    x_r[(3*N_pred)+1:4*N_pred]=yV;
    x_r[(4*N_pred)+1:5*N_pred]=yVr;
    x_r[5*N_pred+1:5*N_pred+20]={pN,q_aN,q_sN,pV,q_aV,q_sV,pVr,q_aVr,q_sVr,ut,mu,gamma_aN,gamma_sN,deltaN,gamma_aV,gamma_sV,deltaV,S0V_anterior,R_a0V_anterior,R_s0V_anterior};
}

// parameters sampling by stan
parameters {
real<lower=0,upper=N0> S0N;
real<lower=0,upper=N0-S0N> I_s0N;
real<lower=0,upper=N0-I_s0N-S0N> I_a0N;
real<lower=0,upper=N0-I_s0N-I_a0N-S0N> R_a0N;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-S0N> R_s0N;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-S0N> D0N;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0N> S0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-S0N> I_s0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-S0N> I_a0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-S0N> R_a0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-S0N>R_s0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N> D0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N-D0V> S0Vr;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N-D0V-S0Vr> I_s0Vr;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N-D0V-S0Vr-I_s0Vr> I_a0Vr;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N-D0V-S0Vr-I_s0Vr-I_a0Vr>R_a0Vr;
//real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N-D0V-S0Vr-I_s0Vr-I_a0Vr-R_a0Vr> D0Vr;

real<lower=D0N+D0V,upper=N0-S0N-I_s0N-I_a0N-R_a0N-R_s0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0Vr-I_s0Vr-I_a0Vr-R_a0Vr> D0;

real<lower=0,upper=N0> Ac0;


real<lower=0.01,upper=1.0> beta_sN;
real<lower=0.01,upper=1.0> beta_aN;
real<lower=0.35,upper=0.6> beta_sV;
real<lower=0.35,upper=0.6> beta_aV;
real<lower=0.35,upper=0.6> beta_sVr;
real<lower=0.35,upper=0.6> beta_aVr;


real<lower=gamma_aV,upper=1/3.0> gamma_aVr;
real<lower=gamma_sV,upper=gamma_aVr> gamma_sVr;
real<lower=0.001,upper=deltaV> deltaVr;

real<lower=0,upper=1> sigma; //se usa en la estamación de las cond iniciales
real<lower=0,upper=1> subr;
}

transformed parameters {

real D0Vr=D0-D0N-D0V;
real R_s0Vr=N0-S0N-I_s0N-I_a0N-R_a0N-R_s0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0Vr-I_s0Vr-I_a0Vr-R_a0Vr-D0;

real theta[10] ={ beta_aN, beta_sN, beta_aV, beta_sV, gamma_aVr, gamma_sVr, deltaVr, beta_aVr, beta_sVr, subr};

real z[N,20];

z[1]={S0N,I_a0N,I_s0N,R_a0N,R_s0N,D0N,Ac0,S0V,I_a0V,I_s0V,R_a0V,R_s0V,D0V,D0,S0Vr,I_a0Vr,I_s0Vr,R_a0Vr,R_s0Vr,D0Vr};
z[2:,] = integrate_ode_rk45(dz_dt, z[1], 0, T[2:], theta, x_r, x_i,rel_tol, abs_tol, max_num_steps);
}

model {
 //Priors
 S0N ~ lognormal(log(S0N_anterior),0.25);
 I_a0N ~ lognormal(log(I_a0N_anterior), 0.25);
 I_s0N ~ lognormal(log(I_s0N_anterior), 0.25);
 R_a0N ~ lognormal(log(R_a0N_anterior), 0.25);
 R_s0N ~ lognormal(log(R_s0N_anterior), 0.25);
 D0N  ~ lognormal(log(D0N_anterior), 0.25);



 S0V ~ lognormal(log(S0V_anterior), 0.25);
 I_a0V ~ lognormal(log(I_a0V_anterior), 0.25);
 I_s0V ~ lognormal(log(I_s0V_anterior), 0.25);
 R_a0V ~ lognormal(log(R_a0V_anterior), 0.25);
 //R_s0V ~ lognormal(log(R_s0V_anterior), 0.25);
//D0V  ~ lognormal(log(D0V_anterior), sigma);


 // bloque 2
 S0Vr ~ lognormal(log(S0Vr_anterior), 0.25);
 I_a0Vr ~ lognormal(log(I_a0Vr_anterior), 0.25);
 I_s0Vr ~ lognormal(log(I_s0Vr_anterior), 0.25);
 R_a0Vr ~ lognormal(log(R_a0Vr_anterior), 0.25);
 R_s0Vr ~ lognormal(log(R_s0Vr_anterior), 0.25);
 D0Vr  ~ lognormal(log(D0Vr_anterior), 0.25);


 Ac0 ~ lognormal(log(yIs[1]), sigma);
 D0 ~ lognormal(log(yD[1]), sigma);
 // subreporte
 subr ~ beta(2,2);



  //rates
  beta_sN ~ normal(0.1, 0.50);//0.50
  beta_aN ~ normal(0.08, 0.50);//0.50
  beta_sV ~ normal(0.5, 0.20);
  beta_aV ~ normal(0.5, 0.20);
  beta_sVr ~ normal(0.5, 0.20);
  beta_aVr ~ normal(0.5, 0.20);

  1/gamma_aVr ~ normal(14.0, 15.0);
  1/gamma_sVr ~ normal(14.0, 15.0);
  deltaVr ~ normal(0.1, 0.050);

  sigma ~ cauchy(0.0, 1.0); // 0, 0.2



  yD  ~ lognormal(log(z[:,14]), sigma);
  //yD  ~ normal(z[:,14], sigma);
  yIs ~ lognormal(log(z[:,7]), sigma);
  //yIs ~ normal(z[:,7], sigma);

}

generated quantities {
    real y_pred[N_pred,20];
    y_pred[1,] = {S0N,I_a0N,I_s0N,R_a0N,R_s0N,D0N,Ac0,S0V,I_a0V,I_s0V,R_a0V,R_s0V,D0V,D0,S0Vr,I_a0Vr,I_s0Vr,R_a0Vr,R_s0Vr,D0Vr};
    y_pred[2:,] = integrate_ode_rk45(dz_dt, y_pred[1], 0, T_pred[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);

}
