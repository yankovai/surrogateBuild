from combinatorics_routines import *
from multiprocessing import Pool
import shelve




class Sparse_Grid:
    """
    Interpolates an N dimension function using Lagrange polynomial basis on a
    Smolyak sparse grid using abscissas of the user's choice. 
    """

    def __init__(self,init_args):
        """
        Parameters
        ----------
        f : class
            Function that is to be interpolated.
        N : int
            Dimension of function f. The number of inputs required by f.
        norm1_error_crit : float
            
        """
        self.f = init_args['function']
        self.N = init_args['N']
        self.error_crit1 = init_args['error_crit1'] # mean
        self.error_crit2 = init_args['error_crit2'] # variance
        self.error_crit3 = init_args['error_crit3'] # hierarchical surplus
        self.max_smolyak_level = init_args['max_smolyak_level']
        self.min_smolyak_level = init_args['min_smolyak_level']
        self.quad_type = init_args['quad_type']

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

            # Start post-processing for current smolyak level
            converged = self.__check_convergence(smolyak_level)
            if converged == True:
                print smolyak_level, 
                break
            
            
                
     #   print self._indice_history['surplus']
     #   abscissas.close()

    def __check_convergence(self,smolyak_level):
        """
        """
        # Sample surrogate and get stats
        mean, var = self.surrogate_mean_variance()
        
        if smolyak_level >= self.min_smolyak_level:
            # Maximum hierarchical surplus of new nodes
            maxhs = max(self._indice_history['surplus'][self._number_terms::])
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
        self._number_terms = len(self._indice_history['x'])
        return converged
            
        
        

    def __process_new_nodes(self,new_nodes):
        """
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
        Evaluate surrogate at current level.
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
        xsamp = np.random.multivariate_normal(f.mu,f.dactive_covmatrix,nsamps)
        fxsamp = np.array([self(x) for x in xsamp])
        mu = np.mean(fxsamp)
        var = np.var(fxsamp)
        return mu, var

##    def __richardson_extrap(self,level,r=2.):
##        """
##        """
##
##        p = self.N
##
##        if self.quad_type == 'gp':
##            # Gauss-Patterson
##            evala = lambda l: ((2.**l)*(l**(p-1.)))**(-r/p)* \
##                    np.log((2**l)*(l**(p-1.)))**((p-1.)*r/(p+1.))       
##            a1 = evala(level-1.)
##            a2 = evala(level)
##            print a1, a2
##            return (a1*self._mu[0] - a2*self._mu[1])/(a1-a2)
##        elif self.quad_type == 'cc':
##            # Clenshaw-Curtis
##            raise NotImplementedError
    
    


# Test on a problem        
from problem_function import *
from time import time
f = Problem_Function(dactive=[0,1,2])

init_args = {'function': f,
             'N': 3,
             'error_crit1': 1e-2,
             'error_crit2': 1e-2,
             'error_crit3': 1e-1,
             'max_smolyak_level': 3,
             'min_smolyak_level': 1,
             'quad_type': 'gp'}
                          
S = Sparse_Grid(init_args)
#t_current = time()
S.build_surrogate()

#x = np.random.uniform(-1.,1.,[10,3])
#for xi in x:
#    print S(xi), f(xi)
#print time() - t_current
    





    
