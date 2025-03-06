from MCDMProblem import MCDMProblem
import numpy as np

class StandardSWING(MCDMProblem):
    _basicModel = """
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
            wStar ~ dirichlet(0.01*e);
            kappaStar ~ gamma(0.001, 0.001);
            kappa ~  gamma(0.001, 0.001);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);

                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);   
            }
        }
    """
   
    _basicModelClustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;
        }

        parameters {
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wc[DmC];
            real<lower=0> ksi[DmC];
            simplex[DmC] theta[DmNo];
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];

            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       
        } 

        model {
            ksi ~ gamma(.001, .001);
            kappa ~ gamma(.001, .001);

            for(i in 1:DmC)
                wc[i] ~ dirichlet(0.001*e);

            
            for (i in 1:DmNo) {
                real contribution[DmC];
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);   
            }
        }
    """

    _basicModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> AltNo;
            int<lower=2> AltC;
            matrix[AltNo, CNo] Alt;
        }

        parameters {
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 

            simplex[CNo] wStar;
            real<lower=0> kappaStar;

            vector[AltC] altMu;
        }

        transformed parameters {
            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real soft_z;
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);

            simplex[CNo] Prf_normalized[DmNo];
            for(i in 1:DmNo) {
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       
            }

        }

        model {
            // Prior
            wStar ~ dirichlet(0.01*e);
            kappaStar ~ gamma(.001,.01);
            W ~ dirichlet(kappaStar*wStar);
            kappa ~ gamma(.01,.01);
            altMu ~ normal(0,5);

            for (i in 1:DmNo){
            
                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);
        }
    """

    _correlatedModel = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        }

        parameters {
            vector[CNo] W_eta[DmNo];
            real<lower=0> kappa[DmNo]; 

            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;           
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];
            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);
            
            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        }

        model {
            wStar_eta ~ multi_normal(mu, Sigma);
            kappa ~ gamma(0.001, 0.001);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, 0.1*Sigma);
                
                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }
        }
    """

    _correlatedModelClustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        }

        parameters {
            vector[CNo] W_eta[DmNo];
            real<lower=0> kappa[DmNo]; 

            vector[CNo] wc_eta[DmC];
            real<lower=0> ksi[DmC];
            simplex[DmC] theta[DmNo];
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];
            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       
            
            simplex[CNo] W[DmNo];
            simplex[CNo] wc[DmC];

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);

            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);
        }

        model {
            ksi ~ gamma(0.001, 0.001);
            kappa ~ gamma(0.001, 0.001);

            for (d in 1:DmC)
                wc_eta[d] ~ multi_normal(mu, Sigma);//wc_eta[d] ~ dirichlet(.001*e);//

            for (i in 1:DmNo){
                real contribution[DmC];
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*.01);
                target += log_sum_exp(contribution);

                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);
            }
        }
    """

    _correlatedModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] Prf[DmNo];
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
            real<lower=0> kappa[DmNo]; 

            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;

            vector[AltC] altMu;
        }

        transformed parameters {
            simplex[CNo] Prf_normalized[DmNo];

            for(i in 1:DmNo)
                Prf_normalized[i] = Prf[i] ./ sum(Prf[i]);       

            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);


            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters

            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);

        }

        model {
            wStar_eta ~ multi_normal(mu, Sigma);
            kappaStar ~ gamma(0.001,0.001);
            kappa ~ gamma(0.001, 0.001);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                Prf_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);

        }
    """

    def __init__(self, Prf, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.Prf = np.array(Prf)
        
        super().__init__(alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)


    @property
    def inputData(self):
        data = self._getCommonData()
        data['Prf'] = self.Prf
        data['e'] =  np.ones(self.CNo)

        return data

    @property
    def OriginalModel(self):
        return self.__originalModel

    @property
    def DmNo(self):
        return self.Prf.shape[0]

    @property
    def CNo(self):
        return self.Prf.shape[1]

    def _checkInputData(self):
        assert self.Prf.shape[0] >= 1, "No input"
        assert self.Prf.shape[1] >= 2, "The number of criteria must be more than 2!"

        return True


if __name__ == "__main__":
    
    prf =  np.array([
        [ 7, 6, 4, 9, 5, 8, 1, 3],
        [ 9, 8, 2, 5, 4, 5, 1, 3],
        [ 8, 8, 5, 9, 5, 5, 1, 2],
        [ 8, 9, 2, 8, 1, 8, 2, 2],
        [ 8, 6, 1, 9, 6, 7, 4, 4],
        [ 9, 8, 1, 9, 7, 5, 5, 6],
        ])

    dmNo, cNo = prf.shape
    altNo = 50
    x = np.random.rand(altNo // 2, cNo)
    altMat = np.concatenate([x*10,x])

    opt = {'CriteriaDependence': False, 'Sigma': np.eye(cNo) }

    swing = StandardSWING( Prf=prf*100, opt=opt, alternatives=altMat, alt_sort_number=2)#, dm_cluster_number=2)
    swing.sampling()
    print('Ok')

