functions {
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
