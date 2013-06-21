import numpy as np
from problem_function import Problem_Function

def mc_sampling_analytic(nsamps=100,seed=414):
    """
    Use Monte-Carlo sampling to get the mean and variance of the true function.
    """

    f = Problem_Function([0,1,2,3,4],'cc') # quad_type just dummy variable here
    np.random.seed(seed)
    x = np.random.multivariate_normal(f.dactive_mu,f.dactive_covmatrix,nsamps)
    fvals = np.array([f.evalf_unnormalized_x(xi) for xi in x])
    mc_mean = np.mean(fvals)
    mc_var = np.var(fvals)

    print ("Using Monte-Carlo sampling of the true function with %d samples and \n"
           "seed number %d. \n") %(nsamps, seed)
    print "Mean: %12.5e" %(mc_mean)
    print "Variance: %12.5e \n" %(mc_var)

def sandwich_formula_analytic():
    """
    Sandwich formula to get variance. Also, analytic sensitivities.
    """

    f = Problem_Function([0,1,2,3,4],'cc') # quad_type just dummy variable here
    xsecs = f.dactive_mu
    kmean = f.evalf_unnormalized_x(xsecs)

    dk_dabs1 = -(xsecs[1]*xsecs[2] + xsecs[4]*xsecs[3])/ \
            (xsecs[1]*(xsecs[0] + xsecs[4])**2)
    
    dk_dabs2 = -xsecs[4]*xsecs[3]/(xsecs[1]**2*(xsecs[0] + xsecs[4]))

    dk_dnufis1 = 1./(xsecs[4] + xsecs[0])

    dk_dnufis2 = xsecs[4]/(xsecs[1]*(xsecs[0] + xsecs[4]))

    dk_dscat = (xsecs[0]*xsecs[3] - xsecs[1]*xsecs[2])/ \
           (xsecs[1]*(xsecs[0] + xsecs[4])**2)

    S = np.matrix([dk_dabs1, dk_dabs2, dk_dnufis1, dk_dnufis2, dk_dscat]) 
    sand_var = S*np.matrix(f.dactive_covmatrix)*S.transpose()

    print ("Using the sandwich formula to calculate the variance of k-infinity.\n"
           "The mean value is obtained by evaluating at the mean xsec values. \n")
    print "Mean: %12.5e" %(kmean)
    print "Variance: %12.5e \n" %(sand_var)
    
    print "Analytic sensitivities of k-inf (at mean xsecs) to:"
    print "absorption 1: %12.5e" %(dk_dabs1*xsecs[0]/kmean)
    print "absorption 2: %12.5e" %(dk_dabs2*xsecs[1]/kmean)
    print "nu-fission 1: %12.5e" %(dk_dnufis1*xsecs[2]/kmean)
    print "nu-fission 2: %12.5e" %(dk_dnufis2*xsecs[3]/kmean)
    print "downscatter : %12.5e \n" %(dk_dscat*xsecs[4]/kmean)

def full_sparse_grid(quad_type,nsamps=100,seed=414):
    """
    Build full, 5 dimensional sparse grid for k-inf and calculate mean, variance,
    and sensitivities using central differencing.
    """

    print "Analyze k-inf using sparse grid interpolant over all 5 dimensions."

    from sparse_grid_build import Sparse_Grid
    
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form the sparse grid."
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form the sparse grid."
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    f = Problem_Function([0,1,2,3,4],quad_type)
    sparse_grid_args = {'function': f,
                        'N': 5,
                        'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-6,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_type': quad_type,
                        'quad_data': quad_data}

    kinf_interp = Sparse_Grid(sparse_grid_args)
    kinf_interp.build_surrogate()
    sg_mu, sg_var = kinf_interp.surrogate_mean_variance(nsamps,seed)

    print "\n"
    print ("Sampling sparse grid interpolant using %d samples and "
           "seed number %d. \n") %(nsamps,seed)
    print "Mean: %12.5e" %(sg_mu)
    print "Variance: %12.5e \n" %(sg_var)

    print "Sensitivities of k-inf (at mean xsecs) to:"
    print "(Calculated using interpolant, central differencing with +/- 1%)"

    xsecs = f.dactive_mu
    sg_sensitivities = np.zeros(5)
    for i in range(5):
        tmp = np.copy(xsecs); tmp.resize(1,5)
        tmp[0,i] *= 1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = kinf_interp(tmpx)[0]
        tmp[0,i] *= .99/1.01
        tmpx = f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = kinf_interp(tmpx)[0]
        sg_sensitivities[i] = (f_fwd-f_bkd)/(sg_mu*0.02)

    print "absorption 1: %12.5e" %(sg_sensitivities[0])
    print "absorption 2: %12.5e" %(sg_sensitivities[1])
    print "nu-fission 1: %12.5e" %(sg_sensitivities[2])
    print "nu-fission 2: %12.5e" %(sg_sensitivities[3])
    print "downscatter : %12.5e \n" %(sg_sensitivities[4])

def oned_weights(quad_type):
    """
    Plot weights of 1D components to see which are the most important"
    """
    
    from dimension_reduction import Surrogate
##  import matplotlib.pylab as pl
    print ("Perform anchored-ANOVA decomposition on k-inf and analyze the \n"
           "importance of 1D components.")

    f = Problem_Function([0,1,2,3,4],quad_type)
   
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    surrogate_args =   {'max_weight_frac': 1.0,
                        'diff_var_order': 1e-3}                       
    sparse_grid_args = {'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-6,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_type': quad_type,
                        'quad_data': quad_data}

    kinf = Surrogate(surrogate_args,sparse_grid_args)
    weights = abs(np.array(kinf.dimensions_weight['weight']))
    weights /= sum(weights)

    print "First order Zabaras weights are..."
    for i in range(5):
        print "Dimension: %2d  Weight: %12.5e" %(i,kinf.dimensions_weight['weight'][i])

    print " "
    print "Normalized weights are (%)..."
    for i in range(5):
        print "Dimension: %2d  Weight: %5.2f" %(i,100.*weights[i])
    
##    pl.figure(1, figsize=(6,6))
##
##    # The slices will be ordered and plotted counter-clockwise.
##    labels = '$\Sigma_{a1}$', '$\Sigma_{a2}$', r'$\nu\Sigma_{f1}$', r'$\nu\Sigma_{f2}$', '$\Sigma_{12}$'
##    pl.pie(weights, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
##    pl.title('Relative Weight of 1D anchored-ANOVA Components', bbox={'facecolor':'0.8', 'pad':5})
##    pl.show()

def full_rom(quad_type,nsamps=100,seed=414):
    
    from dimension_reduction import Surrogate

    print ("Building every component of an anchored-ANOVA decomposition to show \n"
           "it can exactly reproduce the true function.")
   
    if quad_type == 'cc':
        print "Using Clenshaw-Curtis abscissas to form sparse grids. \n"
        from Clenshaw_Curtis import cc_data_main
        quad_data = cc_data_main()
    elif quad_type == 'gp':
        print "Using Gauss-Patterson abscissas to form sparse grids. \n"
        from Gauss_Patterson import gp_data_main
        quad_data = gp_data_main()

    surrogate_args =   {'max_weight_frac': 0.0,
                        'diff_var_order': 1e-15}                       
    sparse_grid_args = {'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-6,
                        'max_smolyak_level': 6,
                        'min_smolyak_level': 1,
                        'quad_type': quad_type,
                        'quad_data': quad_data}

    kinf = Surrogate(surrogate_args,sparse_grid_args)
##    kmean,kvar = kinf._get_surrogate_stats(nsamps=nsamps,seed=seed)
    print "\n"
##    print ("Sampling full anchored-ANOVA decomp using %d samples and "
##           "seed number %d. \n") %(nsamps,seed)
##    print "Mean: %12.5e" %(kmean)
##    print "Variance: %12.5e \n" %(kvar)

    print "Sensitivities of k-inf (at mean xsecs) to:"
    print "(Calculated using full anchored-ANOVA decomp, central differencing with +/- 1%)"

    kinf_f = kinf.objective_function
    xsecs = kinf_f.dactive_mu
    sg_sensitivities = np.zeros(5)
    for i in range(5):
        tmp = np.copy(xsecs); tmp.resize(1,5)
        tmp[0,i] *= 1.01
        tmpx = kinf_f.hypercube2parameters_map(tmp,'hypercube')
        f_fwd = kinf(tmpx)[0]
        tmp[0,i] *= .99/1.01
        tmpx = kinf_f.hypercube2parameters_map(tmp,'hypercube')
        f_bkd = kinf(tmpx)[0]
        sg_sensitivities[i] = (f_fwd-f_bkd)/(kinf.surrogate_mean[-1]*0.02)

    print "absorption 1: %12.5e" %(sg_sensitivities[0])
    print "absorption 2: %12.5e" %(sg_sensitivities[1])
    print "nu-fission 1: %12.5e" %(sg_sensitivities[2])
    print "nu-fission 2: %12.5e" %(sg_sensitivities[3])
    print "downscatter : %12.5e \n" %(sg_sensitivities[4])

if __name__ == '__main__':   
    mc_sampling_analytic()
    sandwich_formula_analytic()
    full_sparse_grid('cc')
    oned_weights('cc')
    full_rom('cc')
   # full_sparse_grid('gp')
   # oned_weights('gp')
   # full_rom('gp')
    



