from Models.BWM.StandardBWM import *
import numpy as np
from Visualization.graph import credal_ranking, weight_distribution
#from statsmodels.stats.moment_helpers import corr2cov

a_b =  np.array([
    [ 3, 4, 6, 1, 5, 2, 9, 7],
    [ 1, 2, 8, 4, 5, 3, 9, 6],
    [ 2, 2, 3, 1, 5, 5, 9, 8],
    [ 2, 1, 8, 2, 9, 3, 8, 8],
    [ 2, 4, 9, 1, 4, 3, 5, 5],
    [ 1, 2, 9, 1, 3, 5, 5, 4],
    ])

a_w =  np.array([
    [ 7, 6, 4, 9, 5, 8, 1, 3],
    [ 9, 8, 2, 5, 4, 5, 1, 3],
    [ 8, 8, 5, 9, 5, 5, 1, 2],
    [ 8, 9, 2, 8, 1, 8, 2, 2],
    [ 8, 6, 1, 9, 6, 7, 4, 4],
    [ 9, 8, 1, 9, 7, 5, 5, 6],
    ])

# A_B = np.array(
#     [[1, 0.20, 0.11, 0.37, 0.28, 0.32],
#      [1, 0.28, 0.07, 0.40, 0.17, 0.61],
#      [1, 0.26, 0.19, 0.58, 0.33, 0.68]])

# A_W = np.array([
#     [0.11, 0.41, 1, 0.13, 0.18, 0.16],
#     [0.07, 0.34, 1, 0.40, 0.25, 0.60],
#     [0.19, 0.14, 1, 0.58, 0.35, 0.80]])


# A_B_normalized = (A_B.T / np.sum(A_B,1)).T #((1 / A_B).T / np.sum(1 / A_B, 1)).T
# A_W_normalized = ((1 / A_W).T / np.sum(1 / A_W, 1)).T # (A_W.T / np.sum(A_W,1)).T

dmNo, cNo = a_w.shape
altNo = 50
x = np.random.rand(altNo // 2, cNo)
altMat = np.concatenate([x*1,x])

criteria_name = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
bwm = StandardBWM( AB=a_b, AW=a_w)
bwm.sampling()

credal_ranking(bwm.AggregatedWeightSamples.T, criteria_name, "nx.html")
weight_distribution(bwm.AggregatedWeightSamples.T, criteria_name, cols=1, row_based=True)

## Criteria independence 
opt = {'CriteriaDependence': True }
bwm_corr = StandardBWM( AB=a_b, AW=a_w, alternatives=altMat, opt=opt)
bwm_corr.sampling()

## Criteria Dependence with Identity Matrix
opt_cor_identity = {'CriteriaDependence': False, 'Sigma': 1*np.eye(cNo) }
bwm_cor_identity = StandardBWM( AB=a_b, AW=a_w, alternatives=altMat, opt=opt_cor_identity)
bwm_cor_identity.sampling()


## Criteria Depedence with covariance 
opt_cor_cov = {'CriteriaDependence': True }
bwm_cor_cov = StandardBWM( AB=a_b, AW=a_w, alternatives=altMat, opt=opt_cor_cov)
bwm_cor_cov.sampling()

print('Ok')
