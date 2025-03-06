import stan
import numpy as np

bayesianBWM_stan = """
 data { 
       int<lower=2> cNo;
       int<lower=1> dmNo;  
       simplex[9] AB_blf[dmNo, cNo]; 
       simplex[9] AW_blf[dmNo, cNo]; 
       vector<lower=0,upper=1>[cNo] e;
    } 

    parameters { 
       simplex[cNo] W[dmNo]; 
       simplex[cNo] wStar;
       real<lower=0> kappaStar;  

       int<lower=1, upper=9> AB[dmNo, cNo]; 
       int<lower=1, upper=9> AW[dmNo, cNo];    
    } 

    transformed parameters { 
    } 

    model {

      kappaStar ~ gamma(.0001,.001);
      wStar ~ dirichlet(0.01*e);


      for (i in 1:dmNo){

        for(j in 1:cNo){
            target += categorical_lpmf( AB[i,j] | AB_blf[i,j]  );
            target += categorical_lpmf( AW[i,j] | AW_blf[i,j]  );
        }

         W[i] ~ dirichlet(kappaStar*wStar);
         AW[i,:] ~ multinomial(W[i]);

         vector[cNo] wInv;

         wInv = e ./ W[i];
         wInv ./= sum(wInv);
         AB[i,:] ~ multinomial(wInv);        
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

dm_no, c_no = a_b.shape

ab_belief = np.zeros((dm_no,c_no,9))
aw_belief = np.zeros((dm_no,c_no,9))

for i in range(dm_no):
      for j in range(c_no):
            elm = a_b[i,j]
            elm_p1 = np.min((elm+1,9))
            ab_belief[i,j,elm-1] += .5
            ab_belief[i,j,elm_p1-1] += .5 

            elmW = a_w[i,j]
            elmW_p1 = np.min((elmW+1,9))
            aw_belief[i,j,elmW-1] += .5
            aw_belief[i,j,elmW_p1-1] += .5 



bwm_data = {'cNo':c_no,
            'dmNo':dm_no,
            'AB_blf':ab_belief,
            'AW_blf':ab_belief,
            'e': np.ones(c_no)
         }

posterior = stan.build(bayesianBWM_stan, data=bwm_data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10000) 

