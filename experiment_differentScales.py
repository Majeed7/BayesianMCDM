from Models.BWM.GaussianBWM import *
from Models.BWM.IntervalBWM import *
from Models.BWM.StandardBWM import *
from Models.BWM.TriangularBWM import *
from Models.BWM.BetaBWM import *
import numpy as np

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


## Standard BWM 
#bwm = StandardBWM( AB=a_b, AW=a_w)
#bwm.sampling()

## Beta BWM 
bwm = BetaBWM( AB_md=a_b, AB_concentration=10*np.ones(a_w.shape), AW_md=a_w, AW_concentration=10*np.ones(a_w.shape))
bwm.sampling()

## Triangular BWM 
bwm_tri = TriangularBWM( AB_md=a_b, AW_md=a_w)
#bwm_tri.sampling()

## Interval BWM 
bwm_intv = IntervalBWM(AB_l = a_b, AB_h = a_b, AW_l = a_w, AW_h = a_w)
#bwm_intv.sampling()

## Gaussian BWM 
eps = 1
bwm_gauss = GaussianBWM(AB_md = a_b, AB_sigma=eps*np.ones(a_w.shape), AW_md = a_w, AW_sigma = eps*np.ones(a_w.shape), num_samples=1000)
bwm_gauss.sampling()

print('Ok')
