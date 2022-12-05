functions {
    real repartitionExp (real lambda, real x1, real x2){
        return exp(-lambda*x1)-exp(-lambda*x2);
    }
}

data {
    // number of days
    int nDaysCases;
    int nDaysOutcome;

    // number of age group
    int nAge;
    
    // param Poisson law
    real lambda; 

    // slope regression log IFR by age
    real cfr0[nAge];

    // para negativ binomial
    real phi;

    // vaccinated each day by age
    real propVaccinated[nDaysCases, nAge];

    // no vaccinated
    real propNoVaccinated[nDaysCases, nAge];

    // population by age
    vector[nAge] population;

    // Outcome and cases
    int nDaysOutcome[nDaysOutcome, nAge];
    matrix[nDaysCases, nAge] cases;
}


parameters {
    vector<lower=0>  [nAge] RR;
}

transformed parameters {
    vector[nDaysCases] gammaIntegrated;
    real  factorReduction[nDaysCases, nAge];
    matrix[nAge, nDaysCases] o;

    for (t in 1:nDaysCases){
        for (s in 1:t){
            gammaIntegrated[t-s+1] = repartitionExp(lambda, s-1, s);
        }
        for (a in 1:nAge){
            o[a, t] = 0;
            for (s in 1:t){
                o[a, t] += cases[s,a] * gammaIntegrated[s] * (propNoVaccinated[s,a] + propVaccinated[s,a] * RR[a]) * cfr0[a];
            }
        } 
    }
}

model {
    RR ~normal(1,1);

    for (t in 1:nDaysOutcome){
        for (a in 1:nAge){
           outcome[t,a] ~  neg_binomial_2(o[t+nDaysCases-nDaysOutcome, a], phi);
        }
    }
}


generated quantities {
    real meanExp;
    matrix[nAge, nDaysOutcome] OutcomePredicted;
    
    meanExp = 1/lambda;
    for (t in  1:nDaysOutcome){
        for (a in 1:nAge){
            OutcomePredicted[a,t] = o[a, t+nDaysCases-nDaysOutcome];
        }
    }

}