import numpy as np
from problem_function import Problem_Function

def anchored_anova_analysis(quad_type,nsamps=100,seed=414):
    """
    Analyze k-inf using anchored-ANOVA decomposition.
    """
    
    from dimension_reduction import Surrogate
    import matplotlib.pylab as pl
    print ("Perform anchored-ANOVA decomposition on the maximum relative \n"
           "power and analyze the importance of each dimension. \n")

    # Initialize
    f = Problem_Function(range(22))
    np.random.seed(seed)
    mu = f.dactive_mu
    covmatrix = f.dactive_covmatrix
    x = np.random.multivariate_normal(mu,covmatrix,nsamps)
    x = f.hypercube2parameters_map(x,'hypercube')
    fx = np.array([f(xi) for xi in x])
    mean_true = np.mean(fx)  
    var_true = np.var(fx) 

    print ("Use Monte-Carlo sampling to get the variance in the maximum \n"
           "relative power achieved during the transient. Using %d samples \n"
           "and a seed number %d. \n") %(nsamps, seed)

    print "Mean: %11.5e" %(mean_true) 
    print "Variance: %11.5e \n" %(var_true)
       
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    surrogate_args =   {'max_weight_frac': 0.02,  
                        'diff_var_order': 1e-0}                       
    sparse_grid_args = {'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-4,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_data': quad_data}

    print ("Performing dimension reduction. To identify the important dimensions \n"
           "the Zabaras weight is used. Higher order components are built on \n"
           "this. Dimensions whose weights exceed 2% are kept. Starting with \n"
           "22 potential dimensions for this problem. \n")
    
    kth = Surrogate(surrogate_args,sparse_grid_args)
    
    print "The important dimensions are calculated to be ... \n"
    params = ['lambda1',
              'lambda2',
              'lambda3',
              'lambda4',
              'lambda5',
              'lambda6',
              'beta1',
              'beta2',
              'beta3',
              'beta4',
              'beta5',
              'beta6',
              'Lambda',
              'Ah',
              'M_coolant',
              'M_fuel',
              'c_p coolant',
              'c_p fuel',
              'velocity',
              'alpha fuel',
              'alpha coolant',
              'parameter in external reactivity']
    for i in kth.important_dimensions:
        print "Dimension Number: %2d  Description: %s" %(i,params[i])

    print ' '
    print "First order Zabaras weights are..."

    for i in range(22):
        print "Dimension: %2d  Weight: %12.5e" %(i,kth.dimensions_weight['weight'][i])

    print ' '
    num_anova_orders = len(kth.surrogate_var)
    print "Mean and variance of reduced order model after each anchored-ANOVA order. \n"

    for i in range(num_anova_orders):
        print "Order: %1d  Mean: %12.5e  Variance: %12.5e" %(i,kth.surrogate_mean[i], kth.surrogate_var[i])

if __name__ == '__main__':
    anchored_anova_analysis('cc')
