import numpy as np
from Models.AHP.StandardAHP import StandardAHP
from Visualization.graph import credal_ranking, weight_distribution

def GMM(PCM):
    dm_no = len(PCM)
    c_no = PCM[0].shape[0]

    weights = np.zeros((dm_no, c_no))
    aggregated_weight = np.zeros((c_no, 1))

    for i in range(dm_no):
        pcm_normalized = PCM[i] / np.sum(PCM[i], axis=0)
        weights[i] = np.exp(np.mean(np.log(pcm_normalized), axis=1))
        weights[i] /= np.sum(weights[i])

    aggregated_weight = np.exp(np.mean(np.log(weights), axis=0))
    aggregated_weight /= np.sum(aggregated_weight)

    return weights, aggregated_weight


PCM =  np.array([
[
    [1,   3,   5,     4, 7],
    [1/3, 1,   3,     2, 5],
    [1/5, 1/3, 1,   1/2, 3],
    [1/4, 1/2,   2,   1, 3],
    [1/7, 1/5, 1/3, 1/3, 1],
],
[
    [1,     4,   3,    5,  8],
    [1/4,   1,   4,    3,  6],
    [1/3, 1/4,   1,    1,  5],
    [1/5, 1/3,   1,    1,  7],
    [1/8, 1/6, 1/5,  1/7,  1],
],
[
    [1,   1/2,   3,   2,  5],
    [2,     1,   5,   1,  2],
    [1/3, 1/5,   1,   2,  1/2],
    [1/2,   1, 1/2,   1,  5],
    [1/5, 1/2,   2, 1/5,  1], 
],
[
    [1,      3,   5,  2,  6],
    [1/3,    1,   1,  3,  2],
    [1/5,    1,   1,  4,  5],
    [1/2,  1/3, 1/4,  1,  1/2],
    [1/6,  1/2, 1/5,  2,  1],
],
[
    [1,     2, 6,   3,   3],
    [1/2,   1, 2,   5,   4],
    [1/6, 1/2, 1, 1/2,   1],
    [1/3, 1/5, 2,   1,   5],
    [1/3, 1/4, 1, 1/5,   1],
],
[
    [1,     2,   5,   4, 9],
    [1/2,   1,   3,   2, 6],
    [1/5, 1/3,   1,   1, 2],
    [1/4, 1/2,   1,   1, 3],
    [1/9, 1/6, 1/2, 1/3, 1],
]
])

weights, aggregated_weight = GMM(PCM)

opt = {'CriteriaDependence': False }
ahp = StandardAHP(PCM=PCM, opt=opt)
ahp.sampling()

criteria_name = ['C1', 'C2', 'C3', 'C4', 'C5']

credal_ranking(ahp.AggregatedWeightSamples.T, criteria_name, "nx.html")
weight_distribution(ahp.AggregatedWeightSamples.T, criteria_name, row_based=False)

print("Done!")

