from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np



def credal_ranking(weights, criteria_name, file_location):
    avg_weights = np.mean(weights, axis=0)
    index = np.argsort(-avg_weights)

    assert len(criteria_name) == weights.shape[1], "Invalid number of criteria"

    sample_no, c_no = weights.shape
    for i in range(c_no):
        criteria_name[i] = criteria_name[i] + ' - ' + str(round(avg_weights[i],3))

    probs=np.empty((c_no, c_no))
    for i in range(c_no):
        for j in range(i, c_no):
            probs[i,j] = round((weights[:,i] >= weights[:,j]).sum() / sample_no,2)
            probs[j,i] = 1 - probs[i,j]

    ## Visualization using pyvis
    net= Network(notebook=False, layout=None, height='800px', width='600px', directed=True)
    for i in range(c_no):
        net.add_node(str(index[i]), size=max(avg_weights[index[i]]*100,10), 
            title=criteria_name[index[i]], label=criteria_name[index[i]], x=0, y=i*200)

    for i in range(c_no-1):
        net.add_edge(str(index[i]), str(index[i+1]), label=str(probs[index[i],index[i+1]]))
        for j in range(i+2, c_no):
            if probs[index[i], index[j]] < 1 and probs[index[i], index[j]] > 0.5:
                net.add_edge(str(index[i]), str(index[j]), label=str(probs[index[i],index[j]]))

    net.toggle_physics(False)
    net.set_edge_smooth("curvedCW")
    #net.show_buttons(filter_=[])
    #net.prep_notebook()

    net.show(file_location)


def weight_distribution(weights, criteria_name, cols=5, row_based=True):
    c_no = len(criteria_name)

    rows =  int(np.ceil(c_no / cols)) # (c_no // cols) + (c_no % cols > 0)

    if not row_based:
        rows, cols = cols, rows 
        fig, axs = plt.subplots(rows,cols, sharey=True, sharex=True, figsize = (3,15))
    else:
        fig, axs = plt.subplots(rows,cols, sharey=True, sharex=True, figsize = (15,3))

    count = 0
    for i in range(rows):
        if count > c_no: break
        for j in range(cols):
            #axs[i,j].hist(weights[:,i], bins = 50)
            #axs[i,j].set_title(criteria_name[i])
            plt.subplot(rows,cols, count+1)
            plt.hist(weights[:,count], bins=50)
            plt.title(criteria_name[count])
            count += 1

    plt.show()

