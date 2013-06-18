import numpy as np
from problem_function import Problem_Function

def anchored_anova_analysis(quad_type,nsamps=100,seed=414):
    """
    Analyze k-inf using anchored-ANOVA decomposition.
    """
    
    from dimension_reduction import Surrogate

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

    surrogate_args =   {'max_weight_frac': 0.02, # 0.02 
                        'diff_var_order': 1e-6}                       
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
    
    print "The important dimensions are calculated to be..."
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
    print "Mean and variance of reduced order model after each anchored-ANOVA order..."

    for i in range(num_anova_orders):
        print "Order: %1d  Mean: %12.5e  Variance: %12.5e" %(i,kth.surrogate_mean[i], kth.surrogate_var[i])
    print " "
    
    # Sensitivity analysis
    print ("Performing sensitivity of true function output to all dimensions... \n"
           "Using central differencing with Delta = 1e-6")
    
    for j in range(22):
        tmp = np.copy(mu)
        tmp[j] *= 1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = f(tmpx)
        tmp[j] *= .99/1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = f(tmpx)
        sensitivity = (f_fwd-f_bkd)/(mean_true*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,params[j]) 
    print " "

    print ("Performing sensitivity analysis of 1D anchored-ANOVA components \n"
           "Using central differencing with Delta = 1e-6")

    comps = kth.hdmr_components['fdactive']
    for j in range(22):
        tmp = np.copy(comps[j].f.dactive_mu)
        tmp *= 1.01
        tmpx = comps[j].f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = comps[j].evaluate(tmpx)
        tmp *= .99/1.01
        tmpx = comps[j].f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = comps[j].evaluate(tmpx)
        sensitivity = (f_fwd-f_bkd)/(kth.surrogate_mean[1]*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,params[j]) 
    print " "

    print ("Performing sensitivity analysis of all anchored-ANOVA components. \n"
           "Higher order components were included in the ROM until the \n"
           "variance between successive levels of interpolation changed with \n"
           "a relaitve difference of < 1e-6."
           "Using central differencing with Delta = 1e-6.")

    for j in range(22):
        tmp = np.copy(kth.objective_function.mu)
        tmp[j] *= 1.01
        tmpx = kth.objective_function.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = kth(tmpx)
        tmp[j] *= .99/1.01
        tmpx = kth.objective_function.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = kth(tmpx)
        sensitivity = (f_fwd-f_bkd)/(kth.surrogate_mean[-1]*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,params[j]) 
    print " "

def full_sparse_grid_analysis(quad_type):
    """
    Building sparse grid interpolant for max_power as a function of all
    dimensions that were deemed to be important by the analysis above.
    """

    from sparse_grid_build import Sparse_Grid
    
    print ("Building a sparse grid interpolant for the maximum power as a \n"
           "function of all the dimensions that were deemed to be important \n"
           "in the analysis above (there are 9).")
    
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    dactive = [7,13,14,15,16,17,18,19,21]
    f = Problem_Function(dactive)

    sparse_grid_args = {'function': f,
                        'N': 9,
                        'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-4,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_data': quad_data}

    S = Sparse_Grid(sparse_grid_args)
    S.build_surrogate()

    print " "
    print ("Use Monte-Carlo sampling to get the variance and mean. \n"
           "Using 100 samples and seed #414 (unless the defaults have \n"
           "been changed in the Sparse_Grid class)... \n")

    print "Mean: %11.5e" %(S._surrogate_mean) 
    print "Variance: %11.5e \n" %(S._surrogate_var)

    print ("Performing sensitivity analysis of 9D sparse grid interpolant. \n"
           "Using central differencing with Delta = 1e-6.")

    for i in range(0,9):
        tmp = np.copy(S.f.dactive_mu)
        tmp[i] *= 1.01
        tmpx = S.f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = S(tmpx)
        tmp[i] *= .99/1.01
        tmpx = S.f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = S(tmpx)
        sensitivity = (f_fwd-f_bkd)/(S._surrogate_mean*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e" %(dactive[i],sensitivity)    
           
if __name__ == '__main__':
    anchored_anova_analysis('cc')
    full_sparse_grid_analysis('cc')
