from scipy.integrate import odeint
import numpy as np
from initialize_surrogate import Hypercube

class Problem_Function(Hypercube):
    """
    Contains the function for which a surrogate model is desired along with
    information about the input variables.
    """
    
    def __init__(self,dactive,quad_type):
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
        
        self.d = 5
        self.mu = np.array([1.04e-2,     # 0  absorption 1
                            1.10e-1,     # 1  absorption 2
                            9.00e-3,     # 2  nufission 1
                            1.91e-1,     # 3  nufission 2
                            1.80e-2])    # 4  downscatter
  
        self.std = np.array([9.06e-5,
                             2.31e-4,
                             4.85e-5,
                             8.87e-4,
                             2.18e-4])
        # Correlation matrix for inputs
        self.corrmatrix = np.array([[ 1.00, 0.07,-0.13, 0.02, 0.75],
                                    [ 0.07, 1.00, 0.06, 0.31,-0.07],
                                    [-0.13, 0.06, 1.00, 0.33,-0.10],
                                    [ 0.02, 0.31, 0.33, 1.00, 0.01],
                                    [ 0.75,-0.07,-0.10, 0.01, 1.00]])
        # Build covariance matrix for the inputs
        s = self.std
        covmatrix = np.zeros([5,5])
        for i in range(0,self.d):
            for j in range(0,self.d):
                covmatrix[i,j] = self.corrmatrix[i,j]*s[i]*s[j]
        self.covmatrix = covmatrix
        self.dactive = list(dactive)
        self.quad_type = quad_type
        
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

        # Return k-inf
        return (x[1]*x[2] + x[4]*x[3])/(x[1]*(x[0] + x[4]))



