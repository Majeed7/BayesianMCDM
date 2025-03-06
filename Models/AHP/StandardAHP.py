from   Models.MCDMProblem import MCDMProblem 
import numpy as np

class StandardAHP(MCDMProblem):
    _basicModel = """
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
   
    _basicModelClustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
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
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }
        } 

        model {
            for(i in 1:DmC){
                wc[i] ~ dirichlet(0.01*e);
                ksi[i] ~ gamma(.001, .001);
            }
            
            for (i in 1:DmNo) {
                real contribution[DmC];
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);   
            }
        }
    """

    _basicModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
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
            array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);

            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

        }

        model {
            wStar ~ dirichlet(0.01*e);
            kappaStar ~ gamma(.01,.01);

            for (i in 1:DmNo){
                W[i] ~ dirichlet(kappaStar*wStar);
                kappa[i] ~ gamma(.001,.001);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);
        }
    """

    _correlatedModel = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
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
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);
        }

        model {
            wStar_eta ~ multi_normal(mu, Sigma);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, 0.1*Sigma);
                kappa[i] ~ gamma(.001,.001);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }
        }
    """

    _correlatedModelClustering = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
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
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }
            
            simplex[CNo] W[DmNo];
            simplex[CNo] wc[DmC];

            for(i in 1:DmNo)
                W[i] = softmax(W_eta[i]);

            for(i in 1:DmC)
                wc[i] = softmax(wc_eta[i]);
        }

        model {
            
            for (d in 1:DmC){
                wc_eta[d] ~ multi_normal(mu, Sigma);
                ksi ~ gamma(.001, .001);
            }

            for (i in 1:DmNo){
                kappa[i] ~ gamma(.001,.001);

                real contribution[DmC];
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma*.01);
                    //contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }
        }
    """

    _correlatedModelSorting = """
        data {
            int<lower=2> CNo;
            int<lower=1> DmNo;
            vector[CNo] AB[DmNo];
            vector[CNo] AW[DmNo];
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
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);       
            }

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
            kappaStar ~ gamma(.0001,.001);
            wStar_eta ~ multi_normal(mu, Sigma);

            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, 0.1*Sigma);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);  
            }

            for (n in 1:AltNo)
                target += log_sum_exp(soft_z[n]);

        }
    """

    def __init__(self, PCM, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.PCM = np.array(PCM)
        
        super().__init__(alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt)


    @property
    def inputData(self):
        data = self._getCommonData()
        data['PCM'] = self.PCM
        data['e'] =  np.ones(self.CNo)

        return data

    @property
    def OriginalModel(self):
        return self.__originalModel

    @property
    def DmNo(self):
        return self.PCM.shape[0]

    @property
    def CNo(self):
        return self.PCM[0].shape[1]

    def _checkInputData(self):

        return True

