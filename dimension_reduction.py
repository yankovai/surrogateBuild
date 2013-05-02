# Import functions
from sparse_grid_build import Sparse_Grid
from status_messages import Status_Message_Surrogate
from problem_function import *
import numpy as np
from itertools import combinations

class HDMR_Component(Sparse_Grid):
    """
    Creates the orthogonal components in the Cut-HDMR functional decomposition
    described by Markus Holtz in Sparse Grid Quadrature in High Dimensions with
    Applications in Finance and Insurance. Specifically, the components defined
    by Eq. 2.4 are created in this class.
    """

    def __init__(self,dactive,hdmr_components,base_point,init_args):
        """
        Parameters
        ----------
        dactive : integer list
            Specifies the indices of the dimensions which the component is a
            function of. The rest of the dimensions are anchored at the anchor
            point.
        hdmr_components : dictionary
            Contains HDMR components that have already been created. These
            components are used to create subsequent components via the
            definition. Under the entry 'dactive' is stored the active
            dimensions used to create the HDMR component. Under the entry
            'fdactive' is stored in the resulting class instance.
        base_point : float
            The value of the high dimensional function evaluated at the anchor
            point.
        init_args : dictionary
            Contains the inputs needed to built a surrogate for the high
            dimensional functio with dimensions 'dactive' active. Needed to
            initialize the class Sparse_Grid.
        """

        # Create surrogate for function projection onto dactive
        Sparse_Grid.__init__(self,init_args)
        self.build_surrogate()
        
        self.dactive = dactive      
        self.hdmr_components = hdmr_components
        self.base_point = base_point

        # Get subcomponents that must be subtracted from current component
        self._get_subcomponent_indices()

        # Get mean and variance of component
        self._get_component_stats()

    def _get_subcomponent_indices(self):
        """
        Returns
        -------
        subcomponent_indices : list of integer tuples
            Contains the 'dactive' entries for each HDMR component comprising
            the current HDMR component. In Markus Holtz' book (referenced in the
            description of this class, these are the 'v' in the summation in
            Eq. 2.4.
        """
        
        subcomponent_indices = []
        ints_less_than_N = range(1,self.N)
        for int_less_than_N in ints_less_than_N:
            for sub_dactive in combinations(self.dactive,int_less_than_N):
                subcomponent_indices.append(sub_dactive)

        # Make 'subcomponent_indices' a class variable
        self.subcomponent_indices = subcomponent_indices

    def _get_component_stats(self,nsamps=100,seed=414):
        """
        Returns
        -------
        hdmr_component_mean : float
            The mean value of the current HDMR component.
        hdmr_component_var: float
            The variance of the current HDMR component.
        """
        
        # Initialize
        np.random.seed(seed)
        mu = self.f.dactive_mu
        covmatrix = self.f.dactive_covmatrix
        
        # Sample covariance matrix of active dimensions
        x = np.random.multivariate_normal(mu,covmatrix,nsamps)
        # Map parameter domain to hypercube domain
        x = self.f.hypercube2parameters_map(x,'hypercube')
        # Evaluate HDMR component at sampled values
        fvals = np.array([self.evaluate(xi) for xi in x])
        
        # Make 'hdmr_component_mean' and 'hdmr_component_var' class variables
        self.hdmr_component_mean = np.mean(fvals)
        self.hdmr_component_var = np.var(fvals)
                
    def evaluate(self,x):
        """
        Parameters
        ----------
        x : float
            Point in the hypercube at which the function current HDMR component
            will be evaluated.

        Returns
        -------
        hdmr_component_val : float
            The value of the current HDMR component at 'x'. This is the
            replacement for __call__ since it's already inherited. A call to
            __call__ will evaluate the Sparse_Grid surrogate built for this HDMR
            component. 
        """

        # Initialize
        hdmr_component_val = self(x) - self.base_point
        dactive = self.hdmr_components['dactive']
        fdactive = self.hdmr_components['fdactive']

        # Subtract contributions from constituents of current component
        for i in self.subcomponent_indices:
            indx = dactive.index(i)
            xdims = list(dactive[indx])
            xval = np.zeros(self.f.d)
            xval[self.f.dactive] = x
            hdmr_component_val -= fdactive[indx].evaluate(xval[xdims])

        return hdmr_component_val

class Surrogate:
    """
    Builds a reduced-order model for an objective function.
    """

    def __init__(self,surrogate_args,sparse_grid_args):
        """
        Parameters
        ----------
        surrogate_args : dictionary
            Contains parameters used to control convergence of the surrogate
            model for the objective function. Controls the Cut-HDMR methods.
            These parameters are described more below.
            
            Contains
            --------
            max_weight_frac : float
                A real number between 0 and 1 that controls how many dimenions
                will be included when constructing higher order components for
                the surrogate. To decide which dimensions will be included in
                higher-order constructs, the magnitudes of all first-order
                sensitivities are normalized by the maximum first-order
                sensitivity. Then, the normalized sensitivities which are
                greater than 'max_weight_frac' are taken as the important
                dimensions. 
            diff_var_order : float
                Convergence criteria for the Cut-HDMR decomposition. Threshold
                for the variance of two consecutive surrogates orders. For
                example, the algorithm will terminate if the difference between
                the variance of the surrogate consisting of all first-order
                effects (and below) and the surrogate consisting of all second
                -order effects (and below) is less than 'diff_var_order'.
                
        sparse_grid_args : dictionary
            Contains parameters used to control the construction of the
            surrogate models using Smolyak sparse grids. See the description
            for the Sparse_Grid class to see which parameters are included.

        Returns
        -------
        hdmr_components : dictionary
        
            Contains
            --------
            dactive : list of tuples
                Each tuple contains the active dimensions of an HDMR component.
            fdactive : list of class instances
                Contains instances of the class HDMR_Component, where each
                instance corresponds to the active dimensions in dactive.

        base_point : float
            The value of the objective function at the anchor point.
        objective_function : class instance
            An instance of the objective function with all dimensions active.
        surrogate_var : float
            Variance of the surrogate created for the objective function.
        surrogate_mean : float
            Mean of the surrogate created for the objective function.
        """

        # Print status messages
        self.message = Status_Message_Surrogate()
        
        sparse_grid_args['function'] = None
        sparse_grid_args['N'] = None   
        self.sparse_grid_args = sparse_grid_args

        # From surrogate_args
        self.max_weight_frac = surrogate_args['max_weight_frac']
        self.diff_var_order = surrogate_args['diff_var_order']
        
        self.hdmr_components = {'dactive': [], 'fdactive': []}

        # Initialize objective function
        P = Problem_Function(dactive=[])
        objectivef_d = P.d
        # Function evaluated at anchor
        self.base_point = P([])
        self.objective_function = Problem_Function(range(0,objectivef_d))

        # Initialize surrogate parameters
        self.surrogate_var = 0.
        self.surrogate_mean = self.base_point
        self.dimensions_weight = {'dactive': [],'weight': []}

        # Build first-order components
        self._first_order_build()

        # Build higher-order components
        self._higher_order_build()
    
    def _first_order_build(self):
        """
        Builds all first-order Cut-HDMR components using Smolyak sparse grid
        algorithm. The sensitivity of the components with respect to the base-
        point is used to define the important dimensions. See Eq. 41 in the
        paper An Adaptive High-Dimensional Stochastic Model Representation
        Technique for the Solution of Stochastic Partial Differential Equations
        by Xiang Ma and Nicholas Zabaras.
        
        Returns
        -------
        important_dimensions : list of tuples
            Contains the important dimensions as deemed using the sensitivity
            metric defined above and the k-means algorithm. 
        """

        # Initialize
        self.message.build_order(1)
        init_args = self.sparse_grid_args
        init_args['N'] = 1
        hdmr_comps = self.hdmr_components
        base_point = self.base_point
        dtotal = self.objective_function.d
        dim_weight = self.dimensions_weight
            
        for dactive in combinations(range(dtotal),1):
            self.message.build_hdmr_component(dactive)
            f = Problem_Function(dactive)
            init_args['function'] = f
            H = HDMR_Component(dactive,hdmr_comps,base_point,init_args)
            hdmr_comps['dactive'].append(dactive)
            hdmr_comps['fdactive'].append(H)
            dim_weight['dactive'].append(dactive)
            # Make sure to not divide by 0 if that's the value of base point
            if base_point != 0.:
                dim_weight['weight'].append(H.hdmr_component_mean/base_point)
            else:
                dim_weight['weight'].append(H.hdmr_component_mean)

        # Update surrogate mean and variance
        self.surrogate_mean, self.surrogate_var = self._get_surrogate_stats()
        # Print mean and variance
        self.message.order_info(self.surrogate_mean,self.surrogate_var)    

        # Identify imporant dimensions
        weights1 = abs(np.array(dim_weight['weight']))
        weights1 /= max(weights1)
        important_dimensions = []
        for w,i in zip(weights1 > self.max_weight_frac,range(dtotal)):
            if w == True:
                important_dimensions.append(i)

        # Make 'important_dimensions' a class variable
        self.important_dimensions = important_dimensions
          
    def _higher_order_build(self):
        """
        Based on the dimensions included in 'important_dimensions' from the
        function first_order_build, higher order components are constructed in
        the Cut-HDMR expansion. When the convergence criteria set by the
        parameter diff_var_order is reached, the algorithm terminates.
        """

        # Initialize
        important_dimensions = self.important_dimensions
        nimportant_dimensions = len(important_dimensions)
        init_args = self.sparse_grid_args
        hdmr_comps = self.hdmr_components
        base_point = self.base_point
        dim_weight = self.dimensions_weight
        converged = False

        for surrogate_order in range(2,nimportant_dimensions + 1):
            self.message.build_order(surrogate_order)
            init_args['N'] = surrogate_order
            for dactive in combinations(important_dimensions,surrogate_order):
                self.message.build_hdmr_component(dactive)
                f = Problem_Function(dactive)
                init_args['function'] = f
                H = HDMR_Component(dactive,hdmr_comps,base_point,init_args)
                hdmr_comps['dactive'].append(dactive)
                hdmr_comps['fdactive'].append(H)
                dim_weight['dactive'].append(dactive)

                # Calculate weights 
                if self.surrogate_mean != 0.:
                    dim_weight['weight'].append(H.hdmr_component_mean/self.surrogate_mean)
                else:
                    dim_weight['weight'].append(H.hdmr_component_mean)
                
            # Check for convergence
            converged = self._check_convergence()
            if converged == True:
                break
                      
    def _check_convergence(self):
        """
        Checks to see whether construction of the surrogate for the objective
        function is complete.

        Returns
        -------
        convergence : logical
            Returns 'True' if the surrogate is complete and 'False' if the
            surrogate has not reached the convergence criteria. 
        """

        # Calculate mean and variance of current surrogate
        surrogate_mean, surrogate_var = self._get_surrogate_stats()

        # Relative difference in mean between two expansion orders
        dvar = abs((surrogate_var - self.surrogate_var)/self.surrogate_var)

        # Converged?
        if dvar < self.diff_var_order:
            converged = True
        else:
            converged = False

        self.message.order_info(surrogate_mean,surrogate_var) 
        # Update mean and variance
        self.surrogate_mean = surrogate_mean
        self.surrogate_var = surrogate_var

        return converged

    def _get_surrogate_stats(self,nsamps=100,seed=414):
        """
        Parameters
        ----------
        nsamps : int
            Number of times to sample the surrogate to obtain mean and variance
            statistics.
        seed : int
            The seed for which to initiate the random number generator.

        Returns
        -------
        surrogate_mean : float
            The mean value of the surrogate after being sampled nsamps times.
        surrogate_var : float
            The variance of the surrogate after being sampled nsamps times. 
        """

        # Initialize
        mu = self.objective_function.mu
        covmatrix = self.objective_function.covmatrix
        np.random.seed(seed)

        # Sample covariance matrix of active dimensions
        x = np.random.multivariate_normal(mu,covmatrix,nsamps)
        # Map parameter domain to hypercube domain
        x = self.objective_function.hypercube2parameters_map(x,'hypercube')
        # Evaluate surrogate at sampled values 
        fvals = np.array([self(xi) for xi in x])
        
        surrogate_mean = np.mean(fvals)
        surrogate_var = np.var(fvals)

        return surrogate_mean, surrogate_var
            
    def __call__(self,x):
        """
        Parameters
        ----------
        x : float array
            The point at which the surrogate is to be evaluated.
            
        Returns
        -------
        surrogate_val : float
            The value of the surrogate evaluated at point x.
        """

        # Initialize
        surrogate_val = self.base_point
        fdactive = self.hdmr_components['fdactive']
        dactive = self.hdmr_components['dactive']
        
        for h,d in zip(fdactive,dactive):
            surrogate_val += h.evaluate(x[list(d)])

        return surrogate_val


