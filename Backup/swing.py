import numpy as np
import stan 

basicModel = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
            vector<lower=0,upper=1>[CNo] e;
        }

        parameters {
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wStar;
            real<lower=0> kappaStar;
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];

            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       
        } 

        model {
            kappaStar ~ gamma(.001,.01);
            wStar ~ dirichlet(0.001*e);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(0.001,0.001); //lognormal(10,100); 
                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);   
            }
        }
    """

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
                   'Prf':a_w,
                   'e': np.ones(cNo),
                  }

posterior_basic = stan.build(basicModel, data=bwm_data_intrvl, random_seed=1)
fit_basic = posterior_basic.sample(num_chains=3, num_samples=10000) 

wStar_samples = fit_basic['wStar']
wStar = np.mean(wStar_samples, axis=-1)
print("The mean distribution of the aggregated weight is: ", wStar)

