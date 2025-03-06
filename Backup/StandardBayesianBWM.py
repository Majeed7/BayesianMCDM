import stan
import numpy as np

bayesianBWM_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       vector<lower=0,upper=1>[CNo] e;
    } 

    parameters { 
       simplex[CNo] W[DmNo]; 
       simplex[CNo] wStar;
       real<lower=0> kappaStar;     
    } 
    
    model {
      kappaStar ~ gamma(.0001,.001);
      wStar ~ dirichlet(0.01*e);

      for (i in 1:DmNo){
         W[i] ~ dirichlet(kappaStar*wStar);
         AW[i,:] ~ multinomial(W[i]);

         vector[CNo] wInv;

         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
      }     
   } 
"""

bayesianBWMCorrelated_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       vector<lower=0,upper=1>[CNo] e;
       vector[CNo] mu; // mean of unnormalized weight
       cov_matrix[CNo] Sigma;
     } 

    parameters { 
       vector[CNo] W_eta[DmNo]; 
       vector[CNo] wStar_eta;
       real<lower=0> kappaStar;
    } 

    transformed parameters { 
       simplex[CNo] W[DmNo]; 
       simplex[CNo] wStar;

       wStar = softmax(wStar_eta);
       
       for(i in 1:DmNo){
           W[i] = softmax(W_eta[i]);
       }
    } 

    model {

      wStar_eta ~ multi_normal(mu, Sigma);

      kappaStar ~ gamma(.0001,.001);
      //wStar ~ dirichlet(0.01*e);


      for (i in 1:DmNo){
         //W[i] ~ dirichlet(kappaStar*wStar);
         W_eta[i] ~ multi_normal(wStar_eta, 0.01*Sigma);

         AW[i,:] ~ multinomial(W[i]);

         vector[CNo] wInv;

         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
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


altNo = 50
x = np.random.rand(altNo // 2, cNo)
altMat = np.concatenate([x*1,x])

covMat = np.eye(cNo) #np.cov(altMat.T)

bwm_data = {'CNo':cNo,
            'DmNo':dmNo,
            'AB':a_b,
            'AW':a_w,
            'e': np.ones(cNo),
            'mu': np.ones(cNo),
            'Sigma': covMat,           
         }

posterior = stan.build(bayesianBWM_stan, data=bwm_data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10000) 

posterior_corr = stan.build(bayesianBWMCorrelated_stan, data=bwm_data, random_seed=1)
fit_corr = posterior_corr.sample(num_chains=4, num_samples=10000) 

print("Done!")




bayesianBWMLog_stan = """
 data { 
       int<lower=2> cNo;
       int<lower=1> dmNo;  
       //int<lower=1> BOSUM[dmNo];  
       real<lower=0> AB_l[dmNo, cNo]; 
       real<lower=0> AB_h[dmNo, cNo]; 
       real<lower=0> AW_l[dmNo, cNo];
       real<lower=0> AW_h[dmNo, cNo]; 
       //int AB_l[dmNo, cNo]; 
       //int AW_l[dmNo, cNo]; 

       vector<lower=0,upper=1>[cNo] e;
    } 

    parameters { 
       simplex[cNo] W[dmNo]; 
       simplex[cNo] wStar;
       real<lower=0> kappaStar; 
       //real AB[dmNo, cNo];
       //real AW[dmNo, cNo];
       vector<lower=0>[cNo] AB[dmNo];
       vector<lower=0>[cNo] AW[dmNo];

    } 

    transformed parameters { 

       simplex[cNo] AB_trnf[dmNo]; 
       simplex[cNo] AW_trnf[dmNo];

       for(i in 1:dmNo){
          AB_trnf[i] = AB[i,:] ./ sum(AB[i,:]);
          AW_trnf[i] = AW[i,:] ./ sum(AW[i,:]);
       }
    } 

    model {

      target += gamma_lpdf(kappaStar | .0001,.001);
      target += dirichlet_lpdf(wStar | 0.01*e);

      //vector<lower=0>[cNo] awi_normalized;
      //vector<lower=0>[cNo] abi_normalized;

      for (i in 1:dmNo){

         target += dirichlet_lpdf(W[i] | kappaStar*wStar);
         
         //AWi_normalized = AW[i,:] ./ sum(AW[i,:]); 
         target += dirichlet_lpdf(AW_trnf[i] | W[i]);

         //target += dirichlet_lpdf( softmax(AW[i]) | W[i]);

         vector[cNo] wInv;

         wInv = e ./ W[i];
         wInv ./= sum(wInv);

         //ABi_normalized = AB[i] ./ sum(AB[i]);
         target += dirichlet_lpdf(AB_trnf[i] | wInv);

         //target += dirichlet_lpdf( softmax(AB[i]) | wInv); 

         target += multinomial_lpmf(AB_l[i,:] | AB_trnf[i]);

         target += uniform_lpdf( AB[i] | AB_l[i], AB_h[i]);
         target += uniform_lpdf( AW[i] | AW_l[i,:], AW_l[i,:]); 

         //target += multinomial_lpmf(AB_l[i,:] | AB_trnf[i]);
         //target += multinomial_lpmf(AW_l[i,:] | AW_trnf[i]);      
      }
   } 
"""
posterior = stan.build(bayesianBWMLog_stan, data=bwm_data_intrvl, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10000) 

print("")