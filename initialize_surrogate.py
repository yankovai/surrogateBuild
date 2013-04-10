from problem_function import Problem_Function
import numpy as np

class Initialize_Problem(Problem_Function):
    """
    Absorbs function information and spawns additional parameters depending
    on the function information. An initializing routine that sets the stage for
    all proceeding calculations.
    """
    
    def __init__(self):
        """
        Returns
        -------
        quad_type : string
            Determines the absissas used for creating the surrogate model.
            Options are either 'gp' for Gauss-Patterson, or 'cc' for Clenshaw-
            Curtis.
        bounds : integer array
            Determines how large to make the hypercube on which the surrogate
            will be based on. For each dimension, the value of bounds determines
            how many standard deviations from the mean the hypercube will extend.
        """
        # Include convergence criteria and stuff like that later
        Problem_Function.__init__(self)

        self.quad_type = 'gp'
        self.bounds = 6.*np.ones(self.d)

    def evaluate_function(self,x):
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
        
        # Map hypercube to function parameter space
        if self.quad_type == 'gp':
            qmin, qmax = -1., 1.
            dq = 2.
        elif self.quad_type == 'cc':
            qmin, qmax = 0., 1.
            dq = 1.       
        dx = self.bounds*self.std
        xmax = self.mu + dx
        xmin = self.mu - dx
        
        xp = xmin + (xmax - xmin)*(x - qmin)/dq
        return self(xp)

    def eval_function_at_anchor(self):
        """
        Returns
        -------
        self.evaluate_function(anchor) : float
            The function value evaluated at the anchor point. Depending on which
            absissas are used, as determined by 'quad_type', the anchor point is
            chosen to be the mean of the corresponding hypercube. 
        """
        
        if self.quad_type == 'gp':
            anchor = np.zeros(self.d)
        elif self.quad_type == 'cc':
            anchor = .5*np.ones(self.d)

        return self.evaluate_function(anchor)
    



    
