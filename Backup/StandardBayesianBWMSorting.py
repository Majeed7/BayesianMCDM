import stan
import numpy as np

bayesianBWMSorting_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       vector<lower=0,upper=1>[CNo] e;

       int<lower=2> AltNo;
       matrix[AltNo, CNo] Alt;
       int<lower=2> AltC;
       vector<lower=0,upper=1>[AltC] eAlt;
    } 

    parameters { 
       simplex[CNo] W[DmNo];
       simplex[CNo] wStar;
       real<lower=0> kappaStar;
       
       //simplex[altC] eta[altNo];
       vector[AltC] altMu;
       //vector<lower=0>[altC] altSigma;
    } 

    transformed parameters {
        vector[AltNo] v = Alt * wStar;
        v ./= (1-v);
        v = log(v);
        array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
        for (n in 1:AltNo) {
            for (k in 1:AltC) {
                soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);
            }
        }
}

    model {

      kappaStar ~ gamma(.0001,.001);
      wStar ~ dirichlet(0.01*e);
      
      //altSigma ~ lognormal(0, 2);
      //altMu ~ normal(0, 10);
      //eta ~ dirichlet(0.01*eAlt);

      for (i in 1:DmNo){
         W[i] ~ dirichlet(kappaStar*wStar);
         AW[i,:] ~ multinomial(W[i]);

         vector[CNo] wInv;

         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
      }

       for (n in 1:AltNo) {
            target += log_sum_exp(soft_z[n]);
        }

      //vector[altNo] v = alt * wStar;
      //v ./= (1-v);
      //v = log( v );
      
      //real contribution[altC];

      //for(i in 1:altNo){
      //    for(j in 1:altC){
      //        contribution[j] += log(eta[i,j]) + log(normal_lpdf( v[i] | altMu[j], altSigma[j] ));
      //    }
      //    target += log_sum_exp(contribution); 
      //}

      //print(v);  
   } 
"""

correlatedBayesianBWMSorting_stan = """
 data { 
       int<lower=2> CNo;
       int<lower=1> DmNo;  
       int AB[DmNo, CNo]; 
       int AW[DmNo, CNo]; 
       vector<lower=0,upper=1>[CNo] e;

       int<lower=2> AltNo;
       matrix[AltNo, CNo] Alt;
       int<lower=2> AltC;
       vector<lower=0,upper=1>[AltC] eAlt;

       vector[CNo] mu; // mean of unnormalized weight
       cov_matrix[CNo] Sigma;
    } 

    parameters { 
       vector[CNo] W_eta[DmNo]; 
       vector[CNo] wStar_eta;
       real<lower=0> kappaStar;
       
       vector[AltC] altMu;
    }

    transformed parameters { 
       simplex[CNo] W[DmNo]; 
       simplex[CNo] wStar;

       wStar = softmax(wStar_eta);
       
       for(i in 1:DmNo){
           W[i] = softmax(W_eta[i]);
       }
    
        vector[AltNo] v = Alt * wStar;
        v ./= (1-v);
        v = log(v);
        array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
        for (n in 1:AltNo) {
            for (k in 1:AltC) {
                soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);
            }
        }
}

    model {

      kappaStar ~ gamma(.0001,.001);
      wStar_eta ~ multi_normal(mu, Sigma);
      //wStar ~ dirichlet(0.01*e);
      
      //altSigma ~ lognormal(0, 2);
      //altMu ~ normal(0, 10);
      //eta ~ dirichlet(0.01*eAlt);

      for (i in 1:DmNo){
         W_eta[i] ~ multi_normal(wStar_eta, 0.01*Sigma);
         //W[i] ~ dirichlet(kappaStar*wStar);
         
         AW[i,:] ~ multinomial(W[i]);

         vector[CNo] wInv;
         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
      }

       for (n in 1:AltNo) {
            target += log_sum_exp(soft_z[n]);
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
            'AltNo': altNo,
            'Alt': altMat,
            'AltC':2,
            'eAlt': np.ones(2),
            'mu': np.ones(cNo)*.01,
            'Sigma': covMat, 
         }

posterior = stan.build(bayesianBWMSorting_stan, data=bwm_data, random_seed=1)
fit = posterior.sample(num_chains=3, num_samples=10000) 

posterior_corr = stan.build(correlatedBayesianBWMSorting_stan, data=bwm_data, random_seed=1)
fit_corr = posterior_corr.sample(num_chains=3, num_samples=10000) 

soft_z_un = np.mean(fit['soft_z'], axis=2)
soft_z = np.exp(soft_z_un)
sum_soft_z = np.sum(soft_z, axis=1).reshape((altNo,1))
soft_z = np.divide(soft_z, sum_soft_z)
z = np.argmax(soft_z, axis=1)

alt_value = 1 / (1 + np.exp(-fit['v']))

mu_un = np.mean(fit['altMu'], axis=1)
mu = 1 / (1 + np.exp(-mu_un))

print("Done!")
