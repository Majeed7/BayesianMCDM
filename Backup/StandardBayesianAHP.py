import stan
import numpy as np

bayesianAHP_stan = """
         data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            matrix[CNo,CNo] PCM[DmNo];
            vector<lower=0,upper=1>[CNo] e;
         }

         parameters {
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wStar;
            real<lower=0> kappaStar;
         }

         transformed parameters {
            matrix[CNo,CNo] PCM_normalized[DmNo];

            for(i in 1:DmNo) {
                for(j in 1:CNo){
                    PCM_normalized[i][,j] = col(PCM[i],j) ./  sum(col(PCM[i],j));
                }
            }
         } 

         model {
            kappaStar ~ gamma(.001,.001);
            wStar ~ dirichlet(0.01*e);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(.001,.001);

                for(j in 1:CNo){
                    PCM_normalized[i][,j] ~ dirichlet(kappa[i]*W[i]);
                }   
            }
         }
"""

PCM =  np.array([
        [
        [1,   2,   4],
        [1/2, 1,   2],
        [1/4, 1/2, 1]
        ],
        [
         [1,   2,   4],
         [1/2, 1,   2],
         [1/4, 1/2, 1]
        ]
       ])


dmNo = PCM.shape[0]
cNo  = PCM.shape[1]


altNo = 50
x = np.random.rand(altNo // 2, cNo)
altMat = np.concatenate([x*1,x])

covMat = np.eye(cNo) #np.cov(altMat.T)

bwm_data = {'CNo':cNo,
            'DmNo':dmNo,
            'PCM':PCM,
            'e': np.ones(cNo),
            'mu': np.ones(cNo),
            'Sigma': covMat,           
         }

posterior = stan.build(bayesianAHP_stan, data=bwm_data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10000) 

#posterior_corr = stan.build(bayesianBWMCorrelated_stan, data=bwm_data, random_seed=1)
#fit_corr = posterior_corr.sample(num_chains=4, num_samples=10000) 

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