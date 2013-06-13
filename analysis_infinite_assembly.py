import numpy as np
from problem_function import Problem_Function

def mc_sampling_analytic(nsamps=1000,seed=414):
    """
    Use Monte-Carlo sampling to get the mean and variance of the true function.
    """

    f = Problem_Function([0,1,2,3,4])
    np.random.seed(seed)
    x = np.random.multivariate_normal(f.dactive_mu,f.dactive_covmatrix,nsamps)
    fvals = np.array([f.evalf_unnormalized_x(xi) for xi in x])
    mc_mean = np.mean(fvals)
    mc_var = np.var(fvals)

    print ("Using Monte-Carlo sampling of the true function with %d samples and \n"
           "seed number %d. \n") %(nsamps, seed)
    print "Mean: %7.5f" %(mc_mean)
    print "Variance: %7.5f \n" %(mc_var)

def sandwich_formula_analytic():
    """
    Sandwich formula to get variance. Also, analytic sensitivities.
    """

    f = Problem_Function([0,1,2,3,4])
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
    print "Mean: %7.5f" %(kmean)
    print "Variance: %7.5f \n" %(sand_var)
    
    print "Analytic sensitivities of k-inf (at mean xsecs) to:"
    print "absorption 1: %8.5f" %(dk_dabs1*xsecs[0]/kmean)
    print "absorption 2: %8.5f" %(dk_dabs2*xsecs[1]/kmean)
    print "nu-fission 1: %8.5f" %(dk_dnufis1*xsecs[2]/kmean)
    print "nu-fission 2: %8.5f" %(dk_dnufis2*xsecs[3]/kmean)
    print "downscatter : %8.5f \n" %(dk_dscat*xsecs[4]/kmean)

def full_sparse_grid(quad_type,nsamps=1000,seed=414):
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

    f = Problem_Function([0,1,2,3,4])
    sparse_grid_args = {'function': f,
                        'N': 5,
                        'error_crit1': 1e-3,
                        'error_crit2': 1e-3,
                        'error_crit3': 1e-4,
                        'max_smolyak_level': 5,
                        'min_smolyak_level': 4, 
                        'quad_data': quad_data}

    kinf_interp = Sparse_Grid(sparse_grid_args)
    kinf_interp.build_surrogate()
    sg_mu, sg_var = kinf_interp.surrogate_mean_variance(nsamps,seed)

    print "\n"
    print ("Sampling sparse grid interpolant using %d samples and "
           "seed number %d. \n") %(nsamps,seed)
    print "Mean: %8.5f" %(sg_mu)
    print "Variance: %8.5f \n" %(sg_var)

    print "Sensitivities of k-inf (at mean xsecs) to:"
    print "(Calculated using interpolant, central differencing with Delta=.00001)"

    xsecs = f.dactive_mu
    kmean = f.evalf_unnormalized_x(xsecs)
    sg_sensitivities = np.zeros(5)
    for i in range(5):
        xp = np.copy(xsecs); xp[i] += 0.001
        xm = np.copy(xsecs); xm[i] -= 0.001
        # Map values in xm,xp to hypercube
        xp = kinf_interp.f.hypercube2parameters_map(xp,'hypercube')
        xm = kinf_interp.f.hypercube2parameters_map(xm,'hypercube')
        kp = kinf_interp(xp)
        km = kinf_interp(xm)
        sg_sensitivities[i] = (kp-km)*xsecs[i]/(2*.00001*kmean)

    print "absorption 1: %8.5f" %(sg_sensitivities[0])
    print "absorption 2: %8.5f" %(sg_sensitivities[1])
    print "nu-fission 1: %8.5f" %(sg_sensitivities[2])
    print "nu-fission 2: %8.5f" %(sg_sensitivities[3])
    print "downscatter : %8.5f \n" %(sg_sensitivities[4])

def dimension_reduction():
    """
    """
    
mc_sampling_analytic()
sandwich_formula_analytic()
#full_sparse_grid('cc')
#full_sparse_grid('gp')



