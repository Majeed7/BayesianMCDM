import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt


def AlternativeRanking(weight_samples, alts, alt_names, file_location):
    sample_no, _ = weight_samples.shape
    alt_no, _ = alts.shape

    alt_util = np.inner(alts, weight_samples).T
    util_avg = np.round(np.mean(alt_util, axis=0), 3)

    probs = np.zeros((alt_no,alt_no))
    for i in range(alt_no):
        for j in range(i+1, alt_no):
            i_wins = len(np.argwhere(alt_util[:,i] >= alt_util[:,j]))
            probs[i,j] = i_wins / sample_no
            probs[j,i] = 1 - probs[i,j]
    
    probs = np.round(probs, 3)
    index = np.argsort(-util_avg)
    
    net= Network(notebook=False, layout=None, height='1200px', width='800px', directed=True)
    for i in range(alt_no):
        net.add_node(str(index[i]), size=max(util_avg[index[i]],10), 
            title=alt_names[index[i]] + ' - ' + str(util_avg[index[i]]), label=alt_names[index[i]], x=0, y=i*130)

    for i in range(alt_no-1):
        net.add_edge(str(index[i]), str(index[i+1]), label=str(probs[index[i],index[i+1]]))
        for j in range(i+2, alt_no):
            if probs[index[i], index[j]] < 1 and probs[index[i], index[j]] > 0.5:
                net.add_edge(str(index[i]), str(index[j]), label=str(probs[index[i],index[j]]))

    net.toggle_physics(False)
    net.set_edge_smooth("curvedCW")
    #net.show_buttons(filter_=[])
    #net.prep_notebook()

    net.show(file_location)


    return probs, alt_util
        
def AlternativeEvaluation(weight_samples, alts):
    mean_weight = np.mean(weight_samples,axis=0)

    avg_util = np.inner(alts, mean_weight)

    return np.round(avg_util, 3)


def utility_distribution(weights, alts_name, cols=5, row_based=True):
    alt_no = len(alts_name)

    rows = int(np.ceil(alt_no / cols)) #(c_no // cols) + (c_no % cols > 0)

    if not row_based:
        rows, cols = cols, rows 
        fig, axs = plt.subplots(rows,cols, sharey=True, sharex=True, figsize = (3,15))
    else:
        fig, axs = plt.subplots(rows,cols, sharey=True, sharex=True, figsize = (15,3))

    count = 0
    for i in range(rows):
        if count > alt_no: break
        for j in range(cols):
            #axs[i,j].hist(weights[:,i], bins = 50)
            #axs[i,j].set_title(criteria_name[i])
            plt.subplot(rows,cols, count+1)
            plt.hist(weights[:,count], bins=50)
            plt.title(alts_name[count])
            count += 1
    plt.savefig('fig2.png')
    plt.show()


