import numpy as np
from Models.BWM.StandardBWM import *
from Alternative.alternative import *

a_b = [
    [5, 2, 2, 1, 9],
    [5, 1, 1, 1, 4],  
    [5, 4, 1, 3, 4],
    [7, 1, 2, 4, 8],
    [9, 2, 2, 1, 7],
    [5, 3, 3, 5, 1],
    [5, 3, 3, 5, 1],    
    [9, 2, 2, 1, 8],
    [9, 3, 3, 1, 9],
    [9, 3, 4, 1, 6]
    ]

a_w = [
    [2, 4, 4, 9, 1],		
    [1, 7, 4, 5, 5],
    [1, 5, 5, 4, 4],
    [3, 8, 7, 6, 1],
    [1, 7, 7, 9, 2],
    [2, 2, 2, 1, 2],
    [2, 2, 2, 1, 2],
    [1, 5, 5, 9, 2],
    [1, 7, 7, 9, 1],
    [1, 7, 6, 9, 5]
    ]

alts = np.array([
    [42,  0.95, 0.94, 0.83, 1],
    [37,  0.93, 0.87, 0.65, 1],
    [23,  0.92, 0.85, 0.59, 1],
    [808, 0.89, 0.91, 0.76, 1],
    [210, 0.92, 0.88, 0.7,  0],
    [48,  0.89, 0.84, 0.63, 0],
    [18,  0.96, 0.73, 0.29, 0],
    [118, 0.94, 0.79, 0.46, 0],
    [22,  1,    0.62, 0.01, 0],
    [75,  1,    0.65, 0.09, 0],
    [244, 0.96, 0.74, 0.32, 0],
    [278, 0.87, 0.8,  0.52, 0],
    [271, 1,    0.61, 0,    1],
    [265, 0.98, 0.29, 0.01, 0],
])
criteria_name = ['Time', 'Precision', 'Recall', 'Recal+', 'Consistency']
alternative_names = ['AML', 'XMap', 'LogMap', 'LogMapBio', 'POMAP++', 'SANOM', 'LogMapLite', 'FCAMapX', 'DOME', 'ALOD2Vec', 'KEPLER', 'Lily', 'ALIN', 'Holontology']

alts[:,0] = max(alts[:,0]) - alts[:,0]
alts[:,0] /= np.max(alts[:,0]) - np.min(alts[:,0])
to_remove = np.array([0,1,2,7,8,9,4,13])
to_remove = -np.sort(-to_remove)
alts = np.delete(alts, to_remove, axis=0)
[alternative_names.pop(i) for i in to_remove]

alts_col_sum = np.max(alts, axis=0)
alts = alts / alts_col_sum


bwm = StandardBWM(AB=a_b, AW=a_w)
bwm.sampling()

probs, alt_util = AlternativeRanking(bwm.AggregatedWeightSamples.T, alts, alt_names=alternative_names, file_location="alt.html")
avg_util2 = AlternativeEvaluation(bwm.AggregatedWeightSamples.T, alts)
print('Done!')


