def evaluate_sparse_grid(x, ihistory, nterms, quad_data, N):
    """
    """
    
    eps = 2.2204460492503131e-16
    
    # Rename inputs for convenience
    iindices = ihistory['iindices'][0:nterms]
    jindices = ihistory['jindices'][0:nterms]
    surplus = ihistory['surplus'][0:nterms]
    
    if nterms > 0:
        interpolant_val = 0.
        for i,j,s in zip(iindices,jindices,surplus):
            a = 1.
            for d in range(0,N):
                # Barycentric interpolation
                w = quad_data[i[d]].bweights
                xi = quad_data[i[d]].knots 
                dx = x[d] - xi
                tmp = w/(dx + eps)
                a *= tmp[j[d]-1]/sum(tmp)
            a *= s
            interpolant_val += a    
        return interpolant_val    
    else:
        return 0.


    

    
