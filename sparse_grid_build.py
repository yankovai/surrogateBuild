from combinatorics_routines import *
from multiprocessing import Pool
from status_messages import Print_Status_Message
import shelve

class Sparse_Grid:
    """
    Interpolates an N dimension function using Lagrange polynomial basis on a
    Smolyak sparse grid using abscissas of the user's choice. See Sparse Grid
    Quadrature in High Dimensions with Applications in Finance and Insurance by
    Markus Holtz for more information (among other sources). 
    """

    def __init__(self,init_args):
        """
        Parameters
        ----------
        f : class
            Function that is to be interpolated.
        N : int
            Dimension of function f. The number of inputs required by f.
        error_crit1 : float
            As the surrogate is built upon with successive levels, the mean of
            the surrogate is taken at each level. One of the convergence
            criteria is the difference between two successive mean values. This
            variable specifies the threshold desired for convergence.
        error_crit2 : float
            Same as error_crit1 but uses the variance of the surrogate rather
            than the mean.
        error_crit3: float
            At each new smolyak level there will be a number of hierarchical
            surpluses calculated. The maximum surplus is taken and used as a
            metric for convergence. The error criteria here is used to set the
            threshold for the convergence.
        max_smolyak_level : int
            The surrogate for the function of interest will not exceed a smolyak
            interpolation level greater than this value.
        min_smolyak_level : int
            The surrogate for the function of interest will not be based on a
            smolyak interpolation level less than this value.
        quad_type : string
            Specifies the quadrature/interpolation scheme to be used in
            constructing the surrogate. Currently only Gauss-Patterson ('gp') is
            available.

        Returns
        -------
        fevals_unique : float dictionary
            Stores the unique points and corresponding function values used to
            create the surrogate. Gets updated after each smolyak level.
            Intended mainly to handle repeat points that arise in the surrogate
            building process.
        indice_history : float dictionary
            Contains all data accumulated for the surrogate. The 'x' values are
            the sparse grid abscissas, which are determined from iindices and
            jindices. The corresponding function value at the point and
            hierarchical surplus are also stored.
        surrogate_mean : float
            The mean value of the surrogate evaluated at the current smolyak
            level, determined by sampling the surrogate using the given
            covariance matrix.
        surrogate_var : float
            The variance of the surrogate evaluated at the current smolyak
            level, determined by sampling the surrogate using the given
            covariance matrix.
        number_terms : int
            Total number of abscissas used to create the surrogate. The length
            of any of the entries in indice_history.        
        """

        # User input
        self.f = init_args['function']
        self.N = init_args['N']
        self.error_crit1 = init_args['error_crit1'] 
        self.error_crit2 = init_args['error_crit2'] 
        self.error_crit3 = init_args['error_crit3']
        self.max_smolyak_level = init_args['max_smolyak_level']
        self.min_smolyak_level = init_args['min_smolyak_level']
        self.quad_type = init_args['quad_type']
        
        # Global data structures needed for build
        self.message = Print_Status_Message()
        self._fevals_unique = {'x':[],'f':[]}
        self._indice_history = {'iindices':[],
                                'jindices':[],
                                'x':[],
                                'f':[],
                                'surplus':[]}
        self._surrogate_mean = 0.
        self._surrogate_var = 0.
        self._number_terms = 0

    def build_surrogate(self):
        """
        Parameters
        ----------
        Initialize the class.
        
        Returns
        -------
        Creates a surrogate for some desired function using Smolyak sparse grid
        interpolation. The relevant formulae can be found in Sparse Grid
        Quadrature in High Dimensions with Applications in Finance and Insurance
        by Markus Holtz. This code assumes the input variables to the desired
        function are normally distrubuted and correlated. 
        """

        # Initialize surrogate
        max_smolyak_level = self.max_smolyak_level
        N = self.N
        quad_type = self.quad_type 
        if self.quad_type == 'gp':
            self._abscissas = shelve.open('gauss_patterson.dat')
        elif self.quad_type == 'cc':
            raise NotImplementedError
        abscissas = self._abscissas 
        converged = False
        
        # Start building surrogate
        for smolyak_level in range(0,max_smolyak_level+1):
            
            # Enumerate ways to get |i| = N + smolyak_level
            for iindices in enumerate_bins_sum(N,N+smolyak_level)[0]:
                points_that_need2b_evaluated = []
                args = {'d': N, 'bin_levels': [iindices], 'quad_type': quad_type}
                for jindices in enumerate_upto_number_nodes(**args):
                    
                    # Get abscissa value
                    xtemp = []
                    for i,j in zip(iindices,jindices):
                        xtemp.append(abscissas['level'+str(i)]['x'][j-1])
                    points_that_need2b_evaluated.append(xtemp)
                    
                    # Store new abscissa information
                    self._indice_history['x'].append(xtemp)
                    self._indice_history['iindices'].append(iindices)
                    self._indice_history['jindices'].append(jindices)
                    
                # Process 'points_that_need2b_evaluated' using multiprocessing
                self.__process_new_nodes(points_that_need2b_evaluated)

            # Post-processing for current smolyak level
            converged = self.__check_convergence(smolyak_level)
            if converged == True:
                break
            
    def __check_convergence(self,smolyak_level):
        """
        Parameters
        ----------
        smolyak_level : int
            The current smolyak level in the construction of the surrogate.

        Returns
        -------
        converged : boolean
            Using calculated parameters from the current smolyak level, this
            function decides whether the surrogate is converged with respect to
            thresholds specified by the user. First, the code checks if the
            minimum smolyak level has been achieved. If it has, the difference
            between two successive surrogate means/variances is compared to the
            threshold. The maximum hierarchical surplus at the current smolyak
            level is also compared to the user-specified threshold.  
        """

        # Maximum hierarchical surplus of new nodes
        maxhs = max(self._indice_history['surplus'][self._number_terms::])
        # Update total number of terms in surrogate
        self._number_terms = len(self._indice_history['x'])
        # Sample surrogate and get stats
        mean, var = self.surrogate_mean_variance()
        
        
        if smolyak_level >= self.min_smolyak_level:
            dmean = abs(mean - self._surrogate_mean) 
            dvar = abs(var - self._surrogate_var)
            
            if dmean < self.error_crit1 and dvar < self.error_crit2:
                converged = True
            elif maxhs < self.error_crit3:
                converged = True
            else:
                converged = False
        else:
            converged = False
            
        # Update values
        self._surrogate_mean = mean
        self._surrogate_var = var
        
        # Print updates to screen
        self.message.level_info(smolyak_level,mean,var,
                                len(self._fevals_unique['x']),maxhs,converged)
        return converged
            
    def __process_new_nodes(self,new_nodes):
        """
        Parameters
        ----------
        new_nodes : int list
            Contains abscissa points spawned by processing through the current
            smolyak interpolation level. These depend on the interpolation
            scheme used.

        Returns
        -------
        Update of _indice_history, _fevals_unique
            Sorts through new_nodes to determine at which nodes the function
            had already been evaluated. For points at which the function has
            never been evaluated, the multiprocessing module is used to evaluate
            the function. The end result is each point in new_nodes having a
            corresponding function value. The processed points are added to
            _indice_history and _fevals_unique. The hierarhcical surplus is also
            calculated at each new point. 
        """
        
        len_new_nodes = len(new_nodes)
        ftemp = [0.]*len_new_nodes
        points_need_feval = []
        indice_need_feval = []
        
        for point,i in zip(new_nodes,range(len_new_nodes)):
            if point in self._fevals_unique['x']:
                indx = self._fevals_unique['x'].index(point)
                ftemp[i] = self._fevals_unique['f'][indx]
            else:
                points_need_feval.append(np.array(point))
                indice_need_feval.append(i)
                
        # Evaluate f(x) for x never before evaluated (multiprocessing)             
        po = Pool()
        fvals_out = po.map(self.f,points_need_feval)
        po.close()
        po.join()

        # Update
        for i,fi in zip(indice_need_feval,fvals_out):
            ftemp[i] = fi
            self._fevals_unique['x'].append(new_nodes[i])
            self._fevals_unique['f'].append(fi)   
        self._indice_history['f'] += ftemp
        # Calculate hierarchical surplus
        self._indice_history['surplus'] += [fi-self(xi) for fi,xi
                                            in zip(ftemp,new_nodes)]

    def __call__(self,x):
        """
        Parameters
        ----------
        x : float
            The N-dimensional point wat which the surrogate model is evaluated.
            
        Returns
        -------
        interpolant_val : float
            The value of the surrogate at x, evaluated at the highest smolyak
            level calculated. 
        """
        
        n = self._number_terms
        abscissas = self._abscissas
        iindices = self._indice_history['iindices'][0:n]
        jindices = self._indice_history['jindices'][0:n]
        surplus = self._indice_history['surplus'][0:n]
        
        if n > 0:
            interpolant_val = 0.
            for i,j,s in zip(iindices,jindices,surplus):
                a = 1.
                for d in range(0,self.N):
                    if i[d] > 1:   
                        temp = abscissas['level'+str(i[d])]['x']
                        xijs = temp[j[d]-1]
                        temp = np.delete(temp,j[d]-1)
                        a *= np.product((x[d] - temp)/(xijs - temp))
                    else:
                        a *= 1.
                a *= s
                interpolant_val += a    
            return interpolant_val    
        else:
            return 0.

    def surrogate_mean_variance(self,nsamps=100,seed=414):
        """
        Parameters
        ----------
        nsamps : int
            The covariance matrix for the active dimensions will be sampled
            this number of times using numpy. These samples will then be
            evaluated at the surrogate.
        seed : int
            The seed number to use when setting numpy's random number generator.
            This ensures consistent results if the work is to be reproduced or
            if nsamps is to be increased.

        Returns
        -------
        mu : float
            The average returned surrogate value when sampled at nsamp points.
        var : float
            The variance of the returned surrogate values when sampled at nsamp
            points.
        """
        
        np.random.seed(seed)
        xsamp = np.random.multivariate_normal(f.dactive_mu,
                                              f.dactive_covmatrix,
                                              nsamps)
        # Map values in xsamp to hypercube
        xsamp = f.hypercube2parameters_map(xsamp,'hypercube')
         
        fxsamp = np.array([self(x) for x in xsamp])
        mu = np.mean(fxsamp)
        var = np.var(fxsamp)
        return mu, var


    
# Test on a problem        
from problem_function import *
from time import time
f = Problem_Function(dactive=[0,1,2])

init_args = {'function': f,
             'N': 3,
             'error_crit1': 1e-3,
             'error_crit2': 1e-3,
             'error_crit3': 1e-4,
             'max_smolyak_level': 5,
             'min_smolyak_level': 1,
             'quad_type': 'gp'}
                          
S = Sparse_Grid(init_args)
S.build_surrogate()

