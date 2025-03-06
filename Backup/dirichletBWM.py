import numpy as np
import stan 

basicModel = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
            vector<lower=0,upper=1>[CNo] e;
        }

        parameters {
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wStar;
            real<lower=0> kappaStar;
        }

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }
        } 

        model {

            kappaStar ~ gamma(.0001,.001);
            wStar ~ dirichlet(0.01*e);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);

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

bwm_data_intrvl = {'CNo':cNo,
                   'DmNo':dmNo,
                   'AB':a_b,
                   'AW':a_w,
                   'e': np.ones(cNo),
                  }

posterior = stan.build(basicModel, data=bwm_data_intrvl, random_seed=1)
fit = posterior.sample(num_chains=3, num_samples=10000) 

wStar_samples = fit['wStar']
wStar = np.mean(wStar_samples, axis=-1)
print("The mean distribution of the aggregated weight is: ", wStar)