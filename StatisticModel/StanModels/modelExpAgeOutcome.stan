functions {
    real repartitionExp (real lambda, real x1, real x2){
        return exp(-lambda*x1)-exp(-lambda*x2);
    }
}

data {
    // number of age group
    int nAge;

    // number of days
    int nDaysCases;
    int nDaysOutcome;
    int nDaysOutcomePred;
    int nDaysCasesPred;

    // Outcome
    int outcome[nAge, nDaysOutcome];
    matrix[nAge, nDaysCases] cases;
    matrix[nAge, nDaysCasesPred] casesPred;
}


parameters {
    real <lower=0.00000001, upper=0.99> cor[nAge];
    real <lower=0> lambda;
    real <lower=0> phi;
}

transformed parameters {
    vector[nDaysCases] gammaIntegrated;
    matrix[nAge, nDaysCases] convolution;

    for (t in 1:nDaysCases){
        for (s in 1:t){
            gammaIntegrated[t-s+1] = repartitionExp(lambda, s-1, s);
        }
        for (a in 1:nAge){
            convolution[a,t] = dot_product(gammaIntegrated[1:t], cases[a,1:t]);
        }
    }
}

model {
    // Priors
    cor ~ uniform(0.00000001,0.99);
    lambda ~normal(0.1, 0.1);
    phi ~ normal(0,5);

    
    for (t in 1:nDaysOutcome){
        for (a in 1:nAge){
           outcome[a, t] ~  neg_binomial_2(convolution[a, t+(nDaysCases-nDaysOutcome)]*cor[a], phi);
        }
    }
}


generated quantities {
    real meanExp;
    matrix[nAge, nDaysOutcome + nDaysOutcomePred] OutcomePredicted;
    vector[nDaysCasesPred] gammaIntegratedPred;
    matrix[nAge, nDaysCasesPred] convolutionPred;

    meanExp = 1/lambda;

    for (t in 1:nDaysCasesPred){
        for (s in 1:t){
            gammaIntegratedPred[t-s+1] = repartitionExp(lambda, s-1, s);
        }

        for (a in 1:nAge){
            convolutionPred[a,t] = dot_product(gammaIntegratedPred[1:t], casesPred[a,1:t]);
        }
    }

    for (a in 1:nAge){
        for (t in 1:(nDaysOutcome+nDaysOutcomePred)){
                OutcomePredicted[a, t] = convolutionPred[a, t+(nDaysCasesPred-nDaysOutcome-nDaysOutcomePred)]*cor[a];
        }
    }
}