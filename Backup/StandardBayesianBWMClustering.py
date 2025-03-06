import stan
import numpy as np

bayesianBWM_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       int<lower=2> DmC;
       vector<lower=0,upper=1>[CNo] e;
    } 

    parameters { 
       simplex[CNo] W[DmNo];
       simplex[CNo] wc[DmC];
       real<lower=0> ksi[DmC]; 
       simplex[DmC] theta[DmNo];
       
       // parameters for aggrgation
       //simplex[cNo] wStar;
       //real<lower=0> kappaStar;     
    } 

    transformed parameters { 
    } 

    model {

      //kappaStar ~ gamma(.0001,.001);
      //wStar ~ dirichlet(0.01*e);

      real contribution[DmC];

      for (i in 1:DmNo){
         //W[i] ~ dirichlet(kappaStar*wStar);
         
         ksi ~ gamma(.001, .001);
         wc ~ dirichlet(0.01*e);

         for(j in 1:DmC) {
            contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
         }
         target += log_sum_exp(contribution); 
         
         
         AW[i,:] ~ multinomial(W[i]);
         vector[CNo] wInv;
         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
      }
   } 
"""

_correlatedModelClustering = """
    data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       int<lower=2> DmC;
       vector<lower=0,upper=1>[CNo] e;

       vector[CNo] mu; // mean of unnormalized weight
       cov_matrix[CNo] Sigma;
    } 

    parameters { 
       vector[CNo] W_eta[DmNo];
       vector[CNo] wc_eta[DmC];
       real<lower=0> ksi[DmC]; 
       simplex[DmC] theta[DmNo]; 
    } 

    transformed parameters {
       simplex[CNo] W[DmNo];
       simplex[CNo] wc[DmC];

       for(i in 1:DmNo){
           W[i] = softmax(W_eta[i]);
       }

       for(i in 1:DmC){
           wc[i] = softmax(wc_eta[i]);
       }
    }

    model {
      real contribution[DmC];

      for (i in 1:DmNo){
        
         ksi ~ gamma(.001, .001);
         //wc ~ dirichlet(0.1*e);
         wc_eta ~ multi_normal(mu, Sigma);

         for(j in 1:DmC) {
            //contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
            contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*.1);
         }
         target += log_sum_exp(contribution); 
         
         
         AW[i,:] ~ multinomial(W[i]);
         vector[CNo] wInv;
         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
      }
    } 
    """

a_b =  np.array([
       [3, 4, 6, 1, 5, 2, 9, 7],
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
            'DmC': 2,
            'mu': np.ones(cNo),
            'Sigma': covMat,    
         }

posterior = stan.build(bayesianBWM_stan, data=bwm_data, random_seed=1)
fit = posterior.sample(num_chains=3, num_samples=3000) 

posterior_corr = stan.build(_correlatedModelClustering, data=bwm_data, random_seed=1)
fit_corr = posterior.sample(num_chains=3, num_samples=3000) 

print("Cluster centers: \n", np.mean(fit['wc'], axis=2))

print("Done!")

