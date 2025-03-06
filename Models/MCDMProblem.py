from abc import ABC, abstractmethod, abstractproperty
import logging
logger = logging.getLogger('stan')
logger.addHandler(logging.NullHandler())
import stan #_jupyter as stan
import numpy as np
import concurrent.futures
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class MCDMProblem(ABC):
    _basicModel = ""
    _basicModelClustering = ""
    _basicModelSorting = ""
    
    _correlatedModel = ""
    _correlatedModelClustering = ""
    _correlatedModelSorting = ""

    _isCorrelatedModel = False
    _isClusteringRequired = False
    _isSortingRequired = False

    def __init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, opt={}):
        self.Alternatives = alternatives
        self.DmClusterNo = dm_cluster_number
        self.AltSortNo = alt_sort_number
        self.numChains = num_chain
        self.numSamples = num_samples
        self.Options = opt

        #self._isCorrelatedModel = False if (isinstance(self.Alternatives, type(None)) and self.Options.get('CriteriaIndependence') == (False or None)) else True
        if self.Alternatives is not None or self.Options.get('Sigma') is not None:
            self._isCorrelatedModel = True
        if self.Options.get('CriteriaDependence') == False:
            self._isCorrelatedModel = False

        self._isSortingRequired =  True if self.AltSortNo > 0 else False
        self._isClusteringRequired = True if self.DmClusterNo > 0 else False


    @property
    def altNo(self):
        return 0 if isinstance(self.Alternatives, type(None)) else self.Alternatives.shape[0]

    @abstractproperty
    def inputData(self):
        pass

    @abstractproperty
    def DmNo(self):
        pass
    
    @abstractproperty
    def CNo(self):
        pass
    
    @abstractmethod
    def _checkInputData(self):
        pass

    @property
    def Model(self):
        model = 'self._'

        model = model + 'correlatedModel' if self._isCorrelatedModel  else model + 'basicModel'
        model = model + 'Clustering' if self._isClusteringRequired else model
        model = model + 'Sorting' if self._isSortingRequired else model 

        print("The used model is: ", model)
        return eval(model)

    def _getCommonData(self):
        data = {}
        data['DmNo'] = self.DmNo
        data['CNo'] = self.CNo
        
        if self.DmClusterNo > 0:
            data['DmC'] = self.DmClusterNo
        
        if not isinstance(self.Alternatives, type(None)):
            data['Alt'] = self.Alternatives
            data['AltNo'] = self.altNo
            
        if self.AltSortNo > 0:
            data['AltC'] = self.AltSortNo
            data['eAlt'] = np.ones(self.AltSortNo)

        if self._isCorrelatedModel:
            data['mu'] = 0.01 * np.ones(self.CNo)
            data['Sigma'] = np.cov(self.Alternatives.T) #np.eye(self.CNo) #
            if not isinstance(self.Options.get('Sigma'), type(None)):
                data['Sigma'] = self.Options.get('Sigma')
                assert data['Sigma'].shape == (self.CNo, self.CNo)


        if self.AltSortNo > 0 and isinstance(self.Alternatives, type(None)):
            raise Exception("Alternatives should be given as input for the sorting problem!")

        return data

    def sampling(self):
        if self._checkInputData:
            posterior = stan.build(self.Model, data=self.inputData, random_seed=1)
            self.Samples = posterior.sample(num_chains=self.numChains, num_samples=self.numSamples)

            #posterior = self.exec_async(stan.build, self.Model, data=self.inputData, random_seed=1)
            #self.Samples = self.exec_async(posterior.sample, num_chains=self.numChains, num_samples=self.numSamples)
            self.processSamples() 
        else:
            raise Exception("The input data is not valid")

    def processSamples(self):
        self.DmWeightSamples = self.Samples['W']
        self.DmWeight = np.mean(self.DmWeightSamples, axis=2)

        if self._isClusteringRequired:
            self.ClusterCenterSamples = self.Samples['wc']
            self.ClusterCenters = np.mean(self.ClusterCenterSamples, axis=2)
            self.DmMembershipSamples = self.Samples['theta']
            self.DmMembership = np.mean(self.DmMembershipSamples, axis=2)
            
        elif self._isSortingRequired:
            self.AggregatedWeightSamples = self.Samples['wStar']
            self.AggregatedWeight = np.mean(self.AggregatedWeightSamples, axis=1)
            
            soft_z_un = np.mean(self.Samples['soft_z'], axis=2)
            soft_z = np.exp(soft_z_un)
            sum_soft_z = np.sum(soft_z, axis=1).reshape((self.altNo,1))
            self.AlternativeMembership = np.divide(soft_z, sum_soft_z)
            self.AlternativeSorting = np.argmax(soft_z, axis=1)

            self.AlternativeValues = 1 / (1 + np.exp(-self.Samples['v']))

            mu_un = np.mean(self.Samples['altMu'], axis=1)
            self.SortingCenters = 1 / (1 + np.exp(-mu_un))
        
        else:
            self.AggregatedWeightSamples = self.Samples['wStar']
            self.AggregatedWeight = np.mean(self.AggregatedWeightSamples, axis=1)
    
    def exec_async(func, *args, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result()

        

     
