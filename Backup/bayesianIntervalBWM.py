import stan
import numpy as np

bayesianIntervalBWM_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       vector[CNo] AB_l[DmNo]; //int AB_l[dmNo, cNo]; //
       vector[CNo] AB_h[DmNo]; //int AB_h[dmNo, cNo]; // 
       vector[CNo] AW_l[DmNo]; //int AW_l[dmNo, cNo]; //
       vector[CNo] AW_h[DmNo]; //int AW_h[dmNo, cNo]; // 
       vector<lower=0,upper=1>[CNo] e;
    } 

    parameters { 
       simplex[CNo] W[DmNo];
       real<lower=0> kappa[DmNo]; 
       simplex[CNo] wStar;
       real<lower=0> kappaStar;
       vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
       vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
    } 

    transformed parameters {
       vector<lower=0>[CNo] AB[DmNo]; 
       vector<lower=0>[CNo] AW[DmNo];
       simplex[CNo] AB_normalized[DmNo];
       simplex[CNo] AW_normalized[DmNo];

      for(i in 1:DmNo){
            AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; // inv_logit(AW_trnf[i]);
            AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; // inv_logit(AB_trnf[i]);

            AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); //AB[i] ./ sum(AB[i]); //
            AW_normalized[i] = AW[i] ./ sum(AW[i]);       
      }
    } 

    model {
      kappaStar ~ gamma(.01,.01);
      wStar ~ dirichlet(0.01*e);

      W ~ dirichlet(kappaStar*wStar);
      kappa ~ gamma(.01,.01);
         
     
      for (i in 1:DmNo){
         
         AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
         AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
      }
   } 
"""
_basicModel = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_l[DmNo]; 
            vector[CNo] AB_h[DmNo]; 
            vector[CNo] AW_l[DmNo]; 
            vector[CNo] AW_h[DmNo];  
            vector<lower=0,upper=1>[CNo] e;
        } 

        parameters { 
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 
            simplex[CNo] wStar;
            real<lower=0> kappaStar;
            vector<lower=0, upper=1>[CNo] AW_trnf[DmNo];
            vector<lower=0, upper=1>[CNo] AB_trnf[DmNo];
        } 

        transformed parameters {
            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AW[i] = AW_l[i] + (AW_h[i] - AW_l[i]) .* AW_trnf[i]; 
                AB[i] = AB_l[i] + (AB_h[i] - AB_l[i]) .* AB_trnf[i]; 

                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }
        } 

        model {
            kappaStar ~ gamma(.01,.01);
            wStar ~ dirichlet(0.01*e);

            W ~ dirichlet(kappaStar*wStar);
            kappa ~ gamma(.01,.01);
                            
            for (i in 1:DmNo) {
                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        } 
    """

a_b =  np.array([
       [ 3, 4, 6, 1, 5, 2, 9, 7],
       [ 1, 2, 8, 4, 5, 3, 9, 6],
       [ 2, 2, 3, 1, 5, 5, 9, 8],
       [ 2, 1, 8, 2, 9, 3, 8, 8],
       [ 2, 4, 9, 1, 4, 3, 5, 5],
       [ 1, 2, 9, 1, 3, 5, 5, 4]])

a_w =  np.array([
        [ 7, 6, 4, 9, 5, 8, 1, 3],
        [ 9, 8, 2, 5, 4, 5, 1, 3],
        [ 8, 8, 5, 9, 5, 5, 1, 2],
        [ 8, 9, 2, 8, 1, 8, 2, 2],
        [ 8, 6, 1, 9, 6, 7, 4, 4],
        [ 9, 8, 1, 9, 7, 5, 5, 6]])


dmNo, cNo = a_w.shape

bwm_data_intrvl = {'CNo':cNo,
                   'DmNo':dmNo,
                   'AB_l':a_b,
                   'AB_h':a_b+1,
                   'AW_l':a_w,
                   'AW_h':a_w+1,
                   'e': np.ones(cNo),
                  }

posterior = stan.build(bayesianIntervalBWM_stan, data=bwm_data_intrvl)
fit = posterior.sample(num_chains=4, num_samples=3000) 

print(np.mean(fit["wStar"], axis=1))

dd = fit["AW"][1,1,1:1000]

import matplotlib.pyplot as plt
plt.plot(dd, 'ro')
plt.show()

print("done")







# target += gamma_lpdf(kappaStar | .0001,.001);
#       target += dirichlet_lpdf(wStar | 0.01*e);

#       for (i in 1:dmNo){

#          target += dirichlet_lpdf( W[i] | kappaStar*wStar);
         
#          target += dirichlet_lpdf( AW_normalized[i] | W[i]);

#          target += dirichlet_lpdf( AB_normalized[i] | W[i]);
#       }