from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
#from IPython.core.display import display, HTML


def plot(weights, criteria_name, file_location=None):
    avg_weights = np.mean(weights, axis=0)
    index = np.argsort(-avg_weights)

    
    assert len(criteria_name) == weights.shape[1], "Invalid number of criteria"
    
    if file_location == None: 
        file_location = 'simple_graph.html'
    
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

    #net.show(file_location)

    net.save_graph(file_location)
    display(HTML(filename=file_location))
    #return probs