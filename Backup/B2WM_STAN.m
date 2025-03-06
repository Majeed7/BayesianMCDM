addpath('MatlabStan-2.15.1.0');
addpath('MatlabProcessManager-master');


B2WM_model = {
   'data {'
   '   int<lower=2> cNo; '
   '   int<lower=1> BOSUM; '
   '   vector<lower=0>[cNo] BO;'
   '   vector<lower=0>[cNo] OW;'
   '}'
   'parameters {'
   '   vector<lower=0,upper=1>[cNo] wW;'
   '   vector<lower=0,upper=1>[cNo] wB;'
   '   vector<lower=0,upper=1>[cNo] wStar;'    
   '}'
   'transformed parameters {'
        'vector<lower=0>[cNo] UNwB;'
        'vector<lower=0>[cNo] TwB;'

        'real sumInverse;'
        
        
        'for (i in 1:cNo)'
            'UNwB[i] = 1 / (13*TwB[i]);'
        
        'sumInverse = sum(UNwB);'
        'wB = UNwB / (sumInverse*BOSUM);' 
   '}'
   'model {'   
   '   OW ~ multinomial(wW);'
   '   BO ~ multinomial(TwB);'
   '   wB ~ dirichlet(kappaStar*wStar);'
   '   wW ~ dirichlet(kappaStar*wStar);'
   '   kappaStar ~ gamma(.01,.01);'  
   '}'
};




data = struct('cNo',3,...
              'BOSUM',11,...
              'BO',[1 2 8],...
              'OW',[8 4 1]);

%fit = stan('model_code',risk_model,'data',data);

fit = stan('model_code',B2WM_model,'data',data,'iter',5000);

% Too much info, just want to check whether chains have finished sampling
fit.verbose = false;

print(fit);

omegaSTAR = fit.extract('permuted',true).omegaStar;
mean(omegaSTAR)

%   '   shape ~ uniform(0,1000);'
%   '   rate ~ uniform(0,1000);'
%   '   shapeStar ~ uniform(0,1000);'
%   '   rateStar ~ uniform(0,1000);'

