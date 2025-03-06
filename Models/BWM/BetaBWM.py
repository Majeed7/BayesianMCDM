from ..MCDMProblem import MCDMProblem
from .StandardBWM import StandardBWM
import numpy as np

class BetaBWM(StandardBWM, MCDMProblem):
    
    _basicModel = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_concentration[DmNo]; 
            vector[CNo] AW_concentration[DmNo]; 
            vector<lower=0,upper=1>[CNo] e;
        } 

        parameters {             
            simplex[CNo] wStar;
            real<lower=0> kappaStar;
            
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 
            
            vector<lower=0>[CNo] AW[DmNo];
            vector<lower=0>[CNo] AB[DmNo];

        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            vector<lower=0, upper=1>[CNo] AB_er[DmNo];
            vector<lower=0, upper=1>[CNo] AW_er[DmNo];

            for(i in 1:DmNo) {
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]);
                AW_normalized[i] = AW[i] ./ sum(AW[i]);

                AB_er[i] = AB_md[i] - AB[i] + 0.5;
                AW_er[i] = AW_md[i] - AW[i] + 0.5;        
            }
        }

        model {
            kappaStar ~ gamma(.001,.001);
            wStar ~ dirichlet(0.01*e);
            
            for (i in 1:DmNo){
                kappa[i] ~ gamma(.001,.001);
                W[i] ~ dirichlet(kappaStar*wStar);

                for(j in 1 :CNo){
                    //AB_er[i,j] ~  beta(0.5*AB_concentration[i,j], 0.5*AB_concentration[i,j]);
                    //AW_er[i,j] ~  beta(0.5*AW_concentration[i,j], 0.5*AW_concentration[i,j]);
                    AB_md[i,j] - (AB[i,j] - 0.5) ~  beta(0.5*AB_concentration[i,j], 0.5*AB_concentration[i,j]); 
                    AW_md[i,j] ~ (AB[i,j] - 0.5) + beta(0.5*AW_concentration[i,j], 0.5*AW_concentration[i,j]); 
                }

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        } 
    """

    _basicModelClustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo];
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo];  
            vector<lower=0,upper=1>[CNo] e;

            int<lower=2> DmC;
        } 

        parameters { 
            simplex[CNo] W[DmNo];
            real<lower=0> kappa[DmNo]; 
            
            simplex[CNo] wc[DmC];
            real<lower=0> ksi[DmC]; 
            simplex[DmC] theta[DmNo];

            vector<lower=0>[CNo] AW[DmNo];
            vector<lower=0>[CNo] AB[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }
        }

        model {      
            kappa ~ gamma(.001,.001);
            ksi ~ gamma(.001, .001);
            for(i in 1:DmC)
                wc[i] ~ dirichlet(0.01*e);

            for (i in 1:DmNo){
                real contribution[DmC];
                for(j in 1:DmC)
                     contribution[j] = log(theta[i,j]) + dirichlet_lpdf( W[i] | ksi[j]*wc[j]);
                target += log_sum_exp(contribution);

                for(j in 1:CNo){
                    AB_md[i,j] ~ normal(AB[i,j], AB_sigma[i,j]);
                    AW_md[i,j] ~ normal(AW[i,j], AW_sigma[i,j]);
                }
                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
        }
    } 
    """
    
    _basicModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo];
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo]; 
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

            vector<lower=0>[CNo] AW[DmNo];
            vector<lower=0>[CNo] AB[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }

            vector[AltNo] v = Alt * wStar;
            v ./= (1-v);
            v = log(v);
            array[AltNo, AltC] real<upper=0> soft_z; // log unnormalized clusters
            
            for (n in 1:AltNo)
                for (k in 1:AltC)
                    soft_z[n, k] = -log(AltC) - 0.5 * pow(altMu[k] - v[n],2);        
        }

        model {
            kappaStar ~ gamma(.01,.01);
            wStar ~ dirichlet(0.01*e);

            W ~ dirichlet(kappaStar*wStar);
            kappa ~ gamma(.01,.01);
            
            for (i in 1:DmNo){
                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

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
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo]; 
            vector<lower=0,upper=1>[CNo] e;

            vector[CNo] mu; // mean of unnormalized weight
            cov_matrix[CNo] Sigma;
        } 

        parameters { 
            vector[CNo] W_eta[DmNo]; 
            vector[CNo] wStar_eta;
            real<lower=0> kappaStar;  
            real<lower=0> kappa[DmNo]; 

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
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
            kappa ~ gamma(.01,.01);
                            
            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlatedModelClustering = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo];  
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

            real<lower=0> kappaStar;

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];  
        }

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
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
            kappa ~ gamma(.01,.01);
            ksi ~ gamma(.001, .001);
            for (i in 1:DmC)
                wc_eta[i] ~ multi_normal(mu, Sigma);
 
            for (i in 1:DmNo) {     
                real contribution[DmC]; 
                for(j in 1:DmC)
                    contribution[j] = log(theta[i,j]) + multi_normal_lpdf( W_eta[i] | wc_eta[j], Sigma);
                target += log_sum_exp(contribution);   

                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }
        }
    """

    _correlatedModelSorting = """
        data { 
            int<lower=2> CNo;
            int<lower=1> DmNo;  
            vector[CNo] AB_md[DmNo]; 
            vector[CNo] AW_md[DmNo]; 
            vector[CNo] AB_sigma[DmNo]; 
            vector[CNo] AW_sigma[DmNo]; 
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
            real<lower=0> kappa[DmNo]; 

            vector[AltC] altMu;

            vector<lower=0>[CNo] AB[DmNo]; 
            vector<lower=0>[CNo] AW[DmNo];
        } 

        transformed parameters {
            simplex[CNo] AB_normalized[DmNo];
            simplex[CNo] AW_normalized[DmNo];

            for(i in 1:DmNo){
                AB_normalized[i] = (e ./ AB[i]) ./ sum(e ./ AB[i]); 
                AW_normalized[i] = AW[i] ./ sum(AW[i]);
            }
                   
            simplex[CNo] W[DmNo];
            simplex[CNo] wStar;

            wStar = softmax(wStar_eta);
            
            for(i in 1:DmNo) {
                W[i] = softmax(W_eta[i]);
            }

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
            kappa ~ gamma(.01,.01);
                            
            for (i in 1:DmNo){
                W_eta[i] ~ multi_normal(wStar_eta, Sigma);

                AB_md[i] ~ normal(AB[i], AB_sigma[i]);
                AW_md[i] ~ normal(AW[i], AW_sigma[i]);

                AW_normalized[i] ~ dirichlet(kappa[i]*W[i]);
                AB_normalized[i] ~ dirichlet(kappa[i]*W[i]);    
            }

            for (n in 1:AltNo)
               target += log_sum_exp(soft_z[n]);
        }
    """

    def __init__(self, AB_md, AW_md, AB_concentration, AW_concentration, alternatives = None, dm_cluster_number=-1, alt_sort_number=-1, num_chain=3, num_samples=1000, opt={}):
        self.AB_MD = np.array(AB_md)
        self.AW_MD = np.array(AW_md)
        self.AB_Concentration = np.array(AB_concentration)
        self.AW_Concentration = np.array(AW_concentration)

        MCDMProblem.__init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples,opt)

    @property
    def inputData(self):
        
        data = self._getCommonData()
        data['AW_md'] = self.AW_MD
        data['AW_concentration'] = self.AW_Concentration
        data['AB_md'] = self.AB_MD
        data['AB_concentration'] = self.AB_Concentration
        data['e'] =  np.ones(self.CNo)

        return data
    

    @property
    def DmNo(self):
        return self.AB_MD.shape[0]
    
    @property
    def CNo(self):
        return self.AB_MD.shape[1]

    def _checkInputData(self):
        assert self.AB_MD.shape == self.AB_Sigma.shape, "AB_MD and AB_Sigma must be of the same size!"
        assert self.AW_MD.shape == self.AW_Sigma.shape, "AW_MD and AW_Sigma must be of the same size!"
        assert self.AB_MD.shape == self.AW_MD.shape, "AB and AW (mean and standard deviation) must be of the same size!"

        assert self.AB_l.shape[0] >=1, "No input"
        assert self.AW_l.shape[1] >=2, "The number of criteria must be more than 2!"

        return True


if __name__ == "__main__":
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

    opt = {'CriteriaDependence': False}

    bwm = GaussianBWM(AB_md= a_b, AB_sigma=0.01*np.ones(a_b.shape), AW_md=a_w, AW_sigma=0.01*np.ones(a_w.shape), opt=opt, alternatives=altMat, alt_sort_number=2)
    bwm.sampling()
    print('Ok')
        