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
        real gamma_aN= theta[1];// recover rate for asintomatic people NO vaccinated
        real gamma_sN= theta[2];// recover rate for sintomatic people NO vaccinated
        real deltaN  = theta[3];// infection fatality rate (only for sintomatic people) for NO vaccinated
        real beta_aN = theta[4];// rate of transmission for asintomatic infection  NO vaccinated
        real beta_sN = theta[5];// rate of transmission for sintomatic infection  NO vaccinated

        // second layer
        real gamma_aV= theta[6];// recover rate for asintomatic people vaccinated
        real gamma_sV= theta[7];// recover rate for sintomatic people vaccinated
        real deltaV  = theta[8];// infection fatality rate of people vaccinated (only for sintomatic people)
        real beta_aV = theta[9];//percentage of beta_aN
        real beta_sV = theta[10];//percentage of beta_sN
        real subr = theta[11];//subreporte

        //fixed
        int N_pred   = x_i[1];
        real pN      = x_r[3*N_pred+1];// probability of being asintomatic when NO vaccinated
        real q_aN    = x_r[3*N_pred+2];// rate of reinfection for asintomatic infection NO vaccinated
        real q_sN    = x_r[3*N_pred+3];// rate of reinfection for sintomatic infection  NO vaccinated
        real pV      = x_r[3*N_pred+4];// probability of being asintomatic when vaccinate
        real q_aV    = x_r[3*N_pred+5];// rate of reinfection for asintomatic infectionof vaccinated
        real q_sV    = x_r[3*N_pred+6];// rate of reinfection for sintomatic infection of vaccinated

        real ut      = x_r[3*N_pred+7];// proportion of asintomatic people isolated before transmission of the disease
        real mu      = x_r[3*N_pred+8];// proportion of sintomatic people isolated before transmission of the disease


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

        //function f_V(t)
        real ft =linear_interpolation(t,x_r[1:N_pred],x_r[(N_pred+1):2*N_pred]);//factor de la efectividad acumulada

        real v   =linear_interpolation(t,x_r[1:N_pred],x_r[2*N_pred+1:3*N_pred]); // velocity of vaccinated with 2 shots o unique shot each day

        // number of people alive
        real NT  = (SN + I_aN + I_sN + R_aN + R_sN) + (SV + I_aV + I_sV + R_aV + R_sV);

        // new No vaccinated infected
        real LambdaN = ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV)) * SN / NT;

        // new vaccinated infected
        real LambdaV = (1-ft) * ((1-ut) * (beta_aN * I_aN + beta_aV * beta_aN * I_aV) + (1-mu) * (beta_sN * I_sN + beta_sV * beta_sN * I_sV)) * SV / NT;


        // new vaccinated in the compartment: susceptible vaccinated
        real vSVN = v * SN / (SN + R_aN + R_sN);
        // new vaccinated in the compartment: recovered asintomatico
        real vRNa = v * R_aN / (SN + R_aN + R_sN);
        // new vaccinated in the compartment : recovered sintomatico
        real vRNs = v * R_sN / (SN + R_aN + R_sN);

        return {
        //update first layer
            -LambdaN+q_aN*R_aN+q_sN*R_sN-vSVN,
            pN*LambdaN-gamma_aN*I_aN,
            (1-pN)*LambdaN-(gamma_sN+deltaN)*I_sN,
            gamma_aN*I_aN-q_aN*R_aN-vRNa,
            gamma_sN*I_sN-q_sN*R_sN-vRNs,
            deltaN*I_sN,

            // cumulative sintomatics cases
            subr*((1-pN)*LambdaN+(1-pV)*LambdaV),

            // update second layer
            -LambdaV+q_aV*R_aV+q_sV*R_sV+v,
            pV*LambdaV-gamma_aV*I_aV,
            (1-pV)*LambdaV-(gamma_sV+deltaV)*I_sV,
            gamma_aV*I_aV-q_aV*R_aV,
            gamma_sV*I_sV-q_sV*R_sV,
            deltaV*I_sV,

            deltaN*I_sN+deltaV*I_sV // cumulative deaths
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
    real ye[N_pred]; // ft function from efectividad.py


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
    real<lower=0> S0N_anterior; // S0 final del mes anterior no vacunados
    real<lower=0> D0_anterior;
    real<lower=0> gamma_aN;
    real<lower=0> gamma_sN;
    real<lower=0> deltaN;
    real<lower=0> S0V_anterior;
    real<lower=0> I_a0N_anterior;
    real<lower=0> I_s0N_anterior;
    real<lower=0> R_a0N_anterior;
    real<lower=0> R_s0N_anterior;

}

transformed data { // we use it to solve the ODE
    real x_r[3*N_pred+8];
    int x_i[1];
    real rel_tol = 1e-6;
    real abs_tol = 1e-6;
    real max_num_steps = 1e4;
    x_i[1]=N_pred;
    x_r[1:N_pred]=T_pred;
    x_r[N_pred+1:2*N_pred]=ye;
    x_r[(2*N_pred)+1:3*N_pred]=yV;
    x_r[3*N_pred+1:3*N_pred+8]={pN,q_aN,q_sN,pV,q_aV,q_sV,ut,mu};
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
//real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-S0N>R_s0V;
//real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-R_s0V-S0N> D0V;
real<lower=0,upper=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-S0N> D0V;

real<lower=0,upper=N0> Ac0;


real<lower=0.01,upper=1.0> beta_sN;
real<lower=0.01,upper=1.0> beta_aN;
real<lower=0.35,upper=0.6> beta_sV;
real<lower=0.35,upper=0.6> beta_aV;


//real<lower=1/30.0,upper=1/3.0> gamma_aV;
//real<lower=1/30.0,upper=gamma_aV> gamma_sV;
//real<lower=0.001,upper=deltaN> deltaV;
real<lower=gamma_aN,upper=1/3.0> gamma_aV;
real<lower=gamma_sN,upper=gamma_aV> gamma_sV;
real<lower=0.001,upper=deltaN> deltaV;

real<lower=0,upper=1> sigma; //se usa en la estamación de las cond iniciales
real<lower=0,upper=1> subr;
}

transformed parameters {

real D0=D0N+D0V;
real R_s0V=N0-S0N-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V-I_s0V-I_a0V-R_a0V-D0V;
//real S0N=N0-I_s0N-I_a0N-R_a0N-R_s0N-D0N-S0V_anterior-I_s0V-I_a0V-R_a0V-R_s0V-D0V;
real theta[11] ={gamma_aN, gamma_sN, deltaN, beta_aN, beta_sN, gamma_aV, gamma_sV, deltaV, beta_aV, beta_sV,subr};

real z[N,14];

z[1]={S0N,I_a0N,I_s0N,R_a0N,R_s0N,D0N,Ac0,S0V,I_a0V,I_s0V,R_a0V,R_s0V,D0V,D0};
z[2:,] = integrate_ode_rk45(dz_dt, z[1], 0, T[2:], theta, x_r, x_i,rel_tol, abs_tol, max_num_steps);


}
model {
 //Priors
 S0N ~ lognormal(log(S0N_anterior),sigma);
 I_a0N ~ lognormal(log(I_a0N_anterior), sigma);
 I_s0N ~ lognormal(log(I_s0N_anterior), sigma);
 R_a0N ~ lognormal(log(R_a0N_anterior), sigma);
 R_s0N ~ lognormal(log(R_s0N_anterior), sigma);
 S0V ~ lognormal(log(S0V_anterior), sigma);
 I_a0V/N0 ~ beta(1,4);
 I_s0V/N0 ~ beta(1,4);
 R_a0V/N0 ~ beta(1,4);
 Ac0 ~ lognormal(log(yIs[1]), sigma);
 D0N ~ lognormal(log(D0_anterior), sigma);
 D0 ~ lognormal(log(yD[1]), sigma);
 // subreporte
 subr ~ beta(2,2);

  //rates
  beta_sN ~ normal(0.1, 0.50);//0.50
  beta_aN ~ normal(0.08, 0.50);//0.50
  beta_sV ~ normal(0.5, 0.20);
  beta_aV ~ normal(0.5, 0.20);

  1/gamma_aV ~ normal(14.0, 15.0);
  1/gamma_sV ~ normal(14.0, 15.0);
  deltaV ~ normal(0.1, 0.050);

  sigma ~ cauchy(0.0, 1.0); // 0, 0.2

  yD  ~ lognormal(log(z[:,14]), sigma);
  yIs ~ lognormal(log(z[:,7]), sigma);

}

generated quantities {
    real y_pred[N_pred,14];
    y_pred[1,] = {S0N,I_a0N,I_s0N,R_a0N,R_s0N,D0N,Ac0,S0V,I_a0V,I_s0V,R_a0V,R_s0V,D0V,D0};
    y_pred[2:,] = integrate_ode_rk45(dz_dt, y_pred[1], 0, T_pred[2:], theta, x_r, x_i, rel_tol, abs_tol, max_num_steps);
}
