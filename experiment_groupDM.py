from Models.BWM.StandardBWM import *
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import softmax

df = pd.read_csv('toursim.csv')
dmNo = df.shape[0]
cNo = int(df.shape[1]/2)

a_b = df.values[:,:cNo]
a_w = df.values[:, cNo:]

altNo = 50
x = np.random.rand(altNo // 2, cNo)
altMat = np.concatenate([x*1,x])

cluster_no = 3
## Criteria independence 
opt = {'CriteriaDependence': False }
bwm = StandardBWM( AB=a_b, AW=a_w, opt=opt, dm_cluster_number=cluster_no, num_samples=1000)
bwm.sampling()

weights = bwm.DmWeight
geo_mean = np.exp(np.mean(np.log(weights), axis=1))
clr_weights = np.log(weights / geo_mean[:, None])
kmeans = KMeans(n_clusters=cluster_no, random_state=0).fit(clr_weights)
centers = kmeans.cluster_centers_
centers_compositional = softmax(centers, axis=1)

print('Ok')