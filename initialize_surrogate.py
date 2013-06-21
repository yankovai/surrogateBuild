import numpy as np

class Hypercube:
    """
    Includes routines that should be absorbed by the function that is to be
    interpolated. Mainly, these routines include the ability to map points from
    the hypercube to the function's parameter space. Also, in 'call' the
    function can be evaluated in specified dimensions only, the other dimensions
    taking on the value of the anchor point. 
    """
    
    def __init__(self):
        """
        Parameters
        ----------
        bounds : integer array
            Determines how large to make the hypercube on which the surrogate
            will be based on. For each dimension, the value of bounds determines
            how many standard deviations from the mean the hypercube will extend.
        quad_type : string
            Determines the abscissas used in building the surrogate. Use either
            'gp' for Gauss-Patterson or 'cc' for Clenshaw-Curtis.
        """

        self.bounds = 6.*np.ones(self.d)

    def hypercube2parameters_map(self,x,map2):
        """
        Paramaters
        ----------
        x : float array
            An array of points that live either in a hypercube or in the
            function parameters space. If 'map2' is 'hypercube' then x can be
            any length less than or equal to the dimension of the function
            (self.d). If 'map2' is 'parameters' then x must be of length self.d.
        map2 : string
            Can either be 'hypercube' or 'parameters'. States what space the
            values of x are mapped to.

        Returns
        -------
        mapped x : float array
            If the value of 'map2' is 'hypercube' then the values in x are
            mapped from the function parameter space to the hypercube. If the
            value of 'map2' is 'paramaters' then the values in x are mapped from
            the hypercube to the parameter space.
        """
        
        if self.quad_type == 'gp':
            qmin, qmax = -1., 1.
            dq = 2.
        elif self.quad_type == 'cc':
            qmin, qmax = -1., 1.
            dq = 2.

        if map2 == 'hypercube':
            # Map from paramater space to hypercube
            dactive = self.dactive
            dx = self.bounds[dactive]*self.std[dactive]
            xmax = self.dactive_mu + dx
            xmin = self.dactive_mu - dx
            return qmin + dq*(x-xmin)/(xmax-xmin)
        elif map2 == 'parameters':
            # Map from hypercube to parameter space
            dx = self.bounds*self.std
            xmax = self.mu + dx
            xmin = self.mu - dx
            return xmin + (xmax - xmin)*(x - qmin)/dq
        
    def evalf_normalized_x(self,x):
        """
        Parameters
        ----------
        x : float array
            The point in the hypercube at which the function of interest will be evaluated.

        Returns
        -------
        self(xp) : float
            The point in the hypercube is first mapped to the corresponding
            point in the function parameter space. The function evaluated at the
            mapped point is returned.
        """

        xp = self.hypercube2parameters_map(x,'parameters')
        return self.evalf_unnormalized_x(xp)

    def _dactive_covariance(self):
        """
        Returns
        -------
        dactive_covmatrix : float array
            The reduced covariance matrix for all inputs. This covariance matrix
            only includes the covariances among the dimensions specified in
            dactive. 
        """

        dactive = self.dactive
        N = len(dactive)
        dactive_covmatrix = np.zeros([N,N])
        for i in range(N):
            dactive_covmatrix[i,:] = self.covmatrix[dactive[i],dactive]
        self.dactive_covmatrix = dactive_covmatrix

    def _dactive_mean(self):
        """
        Returns
        -------
        dactive_mu : float array
            The reduced mean array for all inputs. Only includes the means among
            the dimensions specified in dactive. 
        """

        dactive = self.dactive
        self.dactive_mu = self.mu[dactive]
        
    def __call__(self,x):
        """
        Parameters
        ----------
        x : float array
            The value of each non-anchored dimension at which the function is to
            be evaluated. Should have length less than or equal to d.
        
        Returns
        -------
        self.evaluate_function(anchor) : float
            The function value evaluated at x for dimensions specified in x and
            everywhere else the function is evaluated at the anchor point.
            Depending on which absissas are used, as determined by 'quad_type',
            the anchor point is chosen to be the mean of the corresponding
            hypercube. 
        """
        
        if self.quad_type == 'gp':
            anchor = np.zeros(self.d)
        elif self.quad_type == 'cc':
            anchor = np.zeros(self.d)

        anchor[self.dactive] = x
        return self.evalf_normalized_x(anchor)
