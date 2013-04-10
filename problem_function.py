import numpy as np

class Problem_Function:
    """
    """
    def __init__(self):
        self.d = 3
        self.mu = np.array([.25, 1.25, .75])
        self.std = .05*self.mu
        # Correlation matrix for inputs
        self.corrmatrix = np.array([[1.,.7,.3],
                                    [.7,1.,.5],
                                    [.3,.5,1.]])
        # Build covariance matrix for the inputs
        s = self.std
        covmatrix = np.zeros([3,3])
        for i in range(0,self.d):
            for j in range(0,self.d):
                covmatrix[i,j] = self.corrmatrix[i,j]*s[i]*s[j]
        self.covmatrix = covmatrix

    def __call__(self,x):
        return .3*np.power(x[0],3) + np.cos(x[1]) + np.exp(x[2])
        






        
        
