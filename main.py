from dimension_reduction import Surrogate
from Clenshaw_Curtis import cc_data_main

cc_data = cc_data_main()

surrogate_args =   {'max_weight_frac': 1.0,
                    'diff_var_order': 1e-3}                       
sparse_grid_args = {'error_crit1': 1e-3,
                    'error_crit2': 1e-3,
                    'error_crit3': 1e-4,
                    'max_smolyak_level': 5,
                    'min_smolyak_level': 1,
                    'quad_data': cc_data}

S = Surrogate(surrogate_args,sparse_grid_args)

### CALCULATE FUNCTION MEAN, VARIANCE USING MC
### -------------------------------------------
##nsamps = 100
##seed = 414
##
##np.random.seed(seed)
##f = Problem_Function([0,1,2])
##x = np.random.multivariate_normal(f.dactive_mu,f.dactive_covmatrix,nsamps)
##fvals = np.array([f.evalf_unnormalized_x(xi) for xi in x])
##print np.mean(fvals), np.var(fvals)




