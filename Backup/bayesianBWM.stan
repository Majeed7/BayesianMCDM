data { 
       int<lower=2> cNo;
       int<lower=1> dmNo;  
       //int<lower=1> BOSUM[dmNo];  
       int AB[dmNo, cNo]; 
       int AW[dmNo, cNo]; 
       vector<lower=0,upper=1>[cNo] e;
} 

parameters { 
    simplex[cNo] W[dmNo]; 
    simplex[cNo] wStar;
    real<lower=0> kappaStar;     
} 

transformed parameters { 
} 

model {

    kappaStar ~ gamma(.0001,.001);
    wStar ~ dirichlet(0.1*e);


    for (i in 1:dmNo){
        W[i] ~ dirichlet(kappaStar*wStar);
        AW[i,:] ~ multinomial(W[i]);

        vector[cNo] wInv;

        wInv = e ./ W[i];
        wInv ./= sum(wInv);
        AB[i,:] ~ multinomial(wInv);        
    }
} 