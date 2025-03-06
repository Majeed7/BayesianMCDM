import stan
import numpy as np


bayesianGaussianBWM_stan = """
 data { 
       int<lower=2> cNo;
       int<lower=1> dmNo;  
       vector[cNo] AB_md[dmNo]; 
       vector[cNo] AW_md[dmNo]; 
       vector<lower=0,upper=1>[cNo] e;
    } 

    parameters { 
       simplex[cNo] W[dmNo];
       real<lower=0> kappa[dmNo]; 
       simplex[cNo] wStar;
       real<lower=0> kappaStar;
       vector<lower=0>[cNo] AW[dmNo];
       vector<lower=0>[cNo] AB[dmNo];
    } 

    transformed parameters {
       simplex[cNo] AB_normalized[dmNo];
       simplex[cNo] AW_normalized[dmNo];

      for(i in 1:dmNo){
            AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
            AW_normalized[i] = AW[i] ./ sum(AW[i]);
      }
    }

    model {
      kappaStar ~ gamma(.01,.01);
      wStar ~ dirichlet(0.01*e);

      W ~ dirichlet(kappaStar*wStar);
      kappa ~ gamma(.1,.1);
         
     
      for (i in 1:dmNo){
         AB_md[i] ~ normal(AB[i], 1);
         AW_md[i] ~ normal(AW[i], 1);

         AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
         AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
      }
   } 
"""

a_b =  np.array([[3, 4, 6, 1, 5, 2, 9, 7],
       [1, 2, 8, 4, 5, 3, 9, 6],
       [2, 2, 3, 1, 5, 5, 9, 8],
       [2, 1, 8, 2, 9, 3, 8, 8],
       [2, 4, 9, 1, 4, 3, 5, 5],
       [1, 2, 9, 1, 3, 5, 5, 4]])

a_w =  np.array([
      [ 7, 6, 4, 9, 5, 8, 1, 3],
      [ 9, 8, 2, 5, 4, 5, 1, 3],
      [ 8, 8, 5, 9, 5, 5, 1, 2],
      [ 8, 9, 2, 8, 1, 8, 2, 2],
      [ 8, 6, 1, 9, 6, 7, 4, 4],
      [ 9, 8, 1, 9, 7, 5, 5, 6]]) 

dmNo, cNo = a_w.shape

bwm_data_intrvl = {'cNo':cNo,
                   'dmNo':dmNo,
                   'AB_md':a_b,
                   'AW_md':a_w,
                   'e': np.ones(cNo),
                  }

posterior = stan.build(bayesianGaussianBWM_stan, data=bwm_data_intrvl, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000) 

print("done")