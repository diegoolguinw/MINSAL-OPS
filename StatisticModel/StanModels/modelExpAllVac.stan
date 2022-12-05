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
    
    // Exponential parameter
    real lambda; 

    // COR0
    real cor0;

    // Negative Binomial parameter
    real phi;

    // vaccinated each day by age
    real propVaccinated[nDaysCases];

    // no vaccinated
    real propNoVaccinated[nDaysCases];

    // population by age
    int population;

    // Outcomes and cases
    int outcome[nDaysOutcome];
    vector[nDaysCases] cases;
}


parameters {
    real<lower=0> RR;
}

transformed parameters {
    vector[nDaysCases] gammaIntegrated;
    real  factorReduction[nDaysCases];
    vector[nDaysCases] o;

    for (t in 1:nDaysCases){
        for (s in 1:t){
            gammaIntegrated[t-s+1] = repartitionExp(lambda, s-1, s);
        }
        o[t] = 0;
        for (s in 1:t){
            o[t] += cases[s] * gammaIntegrated[s] * (propNoVaccinated[s] + propVaccinated[s]*RR)*cfr0;
        } 
    }
}

model {
    RR ~normal(1,1);

    for (t in 1:nDaysOutcome){
        outcome[t] ~  neg_binomial_2(d[t+nDaysCases-nDaysOutcome], phi);
    }
}

generated quantities {
    real meanExp;
    vector[nDaysOutcome] OutcomePredicted;
    
    meanExp = 1/lambda;
    for (t in  1:nDaysOutcome){
        OutcomePredicted[t] = o[t+nDaysCases-nDaysOutcome];
    }
}