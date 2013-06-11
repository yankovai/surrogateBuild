import numpy as np
from initialize_surrogate import Hypercube

class Problem_Function(Hypercube):
    """
    Contains the function for which a surrogate model is desired along with
    information about the input variables.
    """
    
    def __init__(self,dactive):
        """
        Parameters
        ----------
        d : int
            Number of dimensions in the function. Put another way, the number of
            inputs that go into the function.
        mu : float array
            Mean of all input variables.
        std : float array
            Standard deviation of all input variables.
        corrmatrix : float array
            Correlation matrix for the input variables.
        covmatrix : float array
            Covariance matrix for the input variables.
        dactive : int list
            A list of dimensions [1,2,3,...,k] that are not anchored, meaning
            their values are free to change. All other dimensions are fixed at
            the anchor value.
        """
        
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
        self.dactive = list(dactive)
        
        Hypercube.__init__(self)
        self._dactive_covariance()
        self._dactive_mean()
        
    def evalf_unnormalized_x(self,x):
        """
        Parameters
        ----------
        x : float array
            The point at which the function is to be evaluated.
            
        Returns
        -------
        f(x) : float
            Function value at x.
        """
        
        return 3.*np.power(x[0],3) + np.power(x[1],2)*x[0] - np.power(x[2],2)

