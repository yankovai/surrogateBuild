from sparse_grid_build import Sparse_Grid
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

    def _get_component_stats(self):
        """
        Returns
        -------
        hdmr_component_mean : float
            The mean value of the current HDMR component.
        hdmr_component_var: float
            The variance of the current HDMR component.
        """
        
        hdmr_component_var = self._surrogate_var 
        hdmr_component_mean = self._surrogate_mean - self.base_point
        hdmr_subcomponent = self.hdmr_components['fdactive']
        
        for i in self.subcomponent_indices:
            indx = self.hdmr_components['dactive'].index(i)
            hdmr_component_var -= hdmr_subcomponent[indx].hdmr_component_var
            hdmr_component_mean -= hdmr_subcomponent[indx].hdmr_component_mean

        # Make 'hdmr_component_mean' and 'hdmr_component_var' class variables
        self.hdmr_component_var = hdmr_component_var
        self.hdmr_component_mean = hdmr_component_mean
                
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

        hdmr_component_val = self(x) - self.base_point
        hdmr_subcomponent = self.hdmr_components['fdactive']
        
        for i in self.subcomponent_indices:
            indx = self.hdmr_components['dactive'].index(i)
            hdmr_component_val -= hdmr_subcomponent[indx].evaluate(x)

        return hdmr_component_val

# Base point (function evaluated at anchor)
base_point = Problem_Function(dactive=[])([])

hdmr_components = {'dactive': [], 'fdactive': []}

init_args = {'function': None,
             'N': None,
             'error_crit1': 1e-3,
             'error_crit2': 1e-3,
             'error_crit3': 1e-4,
             'max_smolyak_level': 5,
             'min_smolyak_level': 1,
             'quad_type': 'gp'}


##dactive = [0,1,2]
##subcomponent_indices = []
##ints_less_than_N = range(1,3+1)
##for int_less_than_N in ints_less_than_N:
##    for sub_dactive in combinations(dactive,int_less_than_N):
##        subcomponent_indices.append(sub_dactive)
##        
##        
##var = 0.
##mean = 0.
##for dactive in subcomponent_indices:
##    f = Problem_Function(dactive)
##    init_args['function'] = f
##    init_args['N'] = len(dactive)
##    H = HDMR_Component(dactive,hdmr_components,base_point,init_args)
##    var += H.hdmr_component_var
##    mean += H.hdmr_component_mean
##    hdmr_components['dactive'].append(dactive)
##    hdmr_components['fdactive'].append(H)

class Surrogate:
    """
    """

    def __init__(self):
        """
        """
        
        self.hdmr_components = {'dactive': [], 'fdactive': []}
        # Function evaluated at anchor
        self.base_point = Problem_Function(dactive=[])([])

    def __call__(self,x):
        """
        """
        



    
    
    
    










