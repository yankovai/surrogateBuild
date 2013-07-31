import numpy as np
from problem_function import Problem_Function
import multiprocessing as mp
import pickle

def mc_sampling_analytic(nsamps=500,seed=414):
    """
    Use Monte-Carlo sampling to get the mean and variance of the true function.
    """
    dactive = range(25)
    f = Problem_Function(dactive,'cc') # quad_type just dummy variable here
    np.random.seed(seed)
    x = np.random.multivariate_normal(f.dactive_mu,f.dactive_covmatrix,nsamps)
    x = f.hypercube2parameters_map(x,'hypercube')
    po = mp.Pool()  
    fvals = po.map(f,x)
    po.close()
    po.join()
    #fvals = np.array([f.evalf_unnormalized_x(xi) for xi in x])
    mc_mean = np.mean(fvals)
    mc_var = np.var(fvals)

    print ("Using Monte-Carlo sampling of the true function with %d samples and \n"
           "seed number %d. \n") %(nsamps, seed)
    print "Mean: %12.5e" %(mc_mean)
    print "Variance: %12.5e \n" %(mc_var)

def finite_difference_sensitivities(true_mean):
    """
    """

    f = Problem_Function(range(25),'cc') # quad_type just dummy variable here
    xsec_names = pickle.load(open('target_xsec_names.dat','rb')) 

    # Sensitivity analysis
    print ("Performing sensitivity of actual function output to all dimensions... \n"
           "Using central differencing with perturbations of +/- 1%")
    
    for j in range(25):
        tmp = np.copy(f.mu)
        tmp[j] *= 1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = f(tmpx)
        tmp[j] *= .99/1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = f(tmpx)
        sensitivity = (f_fwd-f_bkd)/(true_mean*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,xsec_names[j].strip('{').strip('}')) 
    print " "

def rom_1D_components_only(quad_type):
    """
    """
    
    from dimension_reduction import Surrogate

    print ("Creating a ROM of k-eff using only 1D components. Will find the \n"
           "mean, variance, and sensitivities using the ROM.")

    f = Problem_Function(range(25),quad_type)
   
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    surrogate_args =   {'max_weight_frac': 1.0,
                        'diff_var_order': 1e-4}                       
    sparse_grid_args = {'error_crit1': 1e-4,
                        'error_crit2': 1e-4,
                        'error_crit3': 1e-4,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_type': quad_type,
                        'quad_data': quad_data}
    
    print ("The mean and variance of the surrogate comprised of all 1D \n"
           "components was obtained using 500 samples and seed number 414.")

    keff = Surrogate(surrogate_args,sparse_grid_args)
    weights = abs(np.array(keff.dimensions_weight['weight']))
    weights /= sum(weights)

    xsec_names = pickle.load(open('target_xsec_names.dat','rb'))

    print "First order Zabaras weights are..."
    for i in range(25):
        print "Dimension: %2d  Weight: %12.5e" %(i,keff.dimensions_weight['weight'][i])

    print " "
    print "Normalized weights are (%)..."
    for i in range(25):
        print "Dimension: %2d  Weight: %5.2f" %(i,100.*weights[i])


    # Sensitivity Analysis
    print ("Performing sensitivity analysis of 1D anchored-ANOVA components \n"
           "Using central differencing with perturbations of +/- 1%")

    comps = keff.hdmr_components['fdactive']
    for j in range(25):
        tmp = np.copy(comps[j].f.dactive_mu); tmp.resize(1,1)
        tmp *= 1.01
        tmpx = comps[j].f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = comps[j].evaluate(tmpx)
        tmp *= .99/1.01
        tmpx = comps[j].f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = comps[j].evaluate(tmpx)
        sensitivity = (f_fwd-f_bkd)/(keff.surrogate_mean[1]*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,xsec_names[j].strip('{').strip('}')) 
    print " "

def rom_all_components(quad_type):
    """
    """

    from dimension_reduction import Surrogate

    print ("Creating a ROM of k-eff using 1D components and higher order \n"
           "components that were calculated to be 'important'. Will find \n"
           "the mean, variance, and sensitivities using the ROM.")
    
    xsec_names = pickle.load(open('target_xsec_names.dat','rb'))
   
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    surrogate_args =   {'max_weight_frac': 0.02, # 0.02 
                        'diff_var_order': 5e-3}  # 1e-6                    
    sparse_grid_args = {'error_crit1': 1e-4,
                        'error_crit2': 1e-4,
                        'error_crit3': 1e-4,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_type': quad_type,
                        'quad_data': quad_data}

    print ("Performing dimension reduction. To identify the 'important' \n"
           "dimensions the Zabaras weight is used. Higher order components are \n"
           "built on this. Dimensions whose weights exceed 2% are kept. \n"
           "Starting with 25 potential dimensions for this problem. \n")
    
    keff = Surrogate(surrogate_args,sparse_grid_args)

    print ' '
    num_anova_orders = len(keff.surrogate_var)
    print "Mean and variance of reduced order model after each anchored-ANOVA order..."

    for i in range(num_anova_orders):
        print "Order: %1d  Mean: %12.5e  Variance: %12.5e" %(i,keff.surrogate_mean[i], keff.surrogate_var[i])
    print " "
    print ("Performing sensitivity analysis of all anchored-ANOVA components. \n"
           "Higher order components of the important dimensions were included \n"
           "in the ROM until the variance between successive levels of \n"
           "interpolation changed with a relative difference of < 5e-3. \n"
           "Using central differencing with perturbations of +/- 1%.")

    for j in range(25):
        tmp = np.copy(keff.objective_function.mu); tmp.resize(1,25)
        tmp[0,j] *= 1.01
        tmpx = keff.objective_function.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = keff(tmpx)[0]
        tmp[0,j] *= .99/1.01
        tmpx = keff.objective_function.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = keff(tmpx)[0]
        sensitivity = (f_fwd-f_bkd)/(keff.surrogate_mean[-1]*0.02)
        print "Dimension: %2d  Sensitivity: %12.5e  Description: %s" %(j,sensitivity,xsec_names[j].strip('{').strip('}')) 
    print " "

if __name__ == '__main__':
    mc_sampling_analytic()
    finite_difference_sensitivities(true_mean=1.15448)
    rom_1D_components_only('cc')
    rom_1D_components_only('gp')
##    rom_all_components('cc')


    



