"""
combinatorics_routines.py: Contains combinatorics routines needed for
constructing Smolyak sparse grids.

Written by : Artem Yankov
Email : yankovai@umich.edu
Date : 4/3/2013
"""

import numpy as np

def enumerate_bins_sum(d,levelsum):
    """
    Parameters
    ----------
    d : int
        The number of bins.
    levelsum : int
        Each bin has a number. The sum of the numbers across all bins will equal
        levelsum.
        
    Returns
    -------
    indices : int array
        All enumerations of the number of ways to distribute integers to d bins
        such that the sum of the numbers across all bins will equal levelsum.
    nindices : int
        The number of ways to arrive at the output indices... the number of
        possible enumerations.

    Example
    -------
    Let d=3 and levelsum=5. The bins can be labelled as b1, b2, b3. We want to
    enumerate all ways to write b1+b1+b3=5. For this case the function will
    return the list,
    [[3, 1, 1], [2, 2, 1], [1, 3, 1], [2, 1, 2], [1, 2, 2], [1, 1, 3]]

    Notes
    -----
    This is Algorithm 4.2 of the book Sparse Grid Quadrature in High Dimensions
    with Applications in Finance and Insurance by Markus Holtz.
    """
    
    if d == 1:
        return [[levelsum]], 1 
    
    # Initialize
    p = 0
    m = levelsum - d + 1 
    k = [0] + [1]*(d - 1)
    kh = [m]*d
    go = 1
    indices = []

    # Outside loop continues until all indices have been enumerated
    while go == 1:
        # Compare two lists k and kh to output valid index
        k[p] = k[p] + 1
        if k[p] > kh[p]:
            if p == (d-1):
                nindices = len(indices)
                return indices, nindices
            k[p] = 1
            p = p + 1
        else:
            for j in range(0,p):
                kh[j] = kh[p] - k[p] + 1
            k[0] = kh[0]
            p = 1
            # Valid index combination
            indices.append(list(k))
       
def enumerate_upto_number_nodes(d,mi):
    """
    Parameters
    ----------
    d : int
        The number of bins, dimensionality of the problem.
    bin_levels : int list
        The output from routine enumerate_bins_sum. The number of ways d bins
        can hold certain integer values such that the sum of their values totals
        levelsum (ref enumerate_bins_sum).
    quad_type: string
        Specifies the quadrature/interpolation type that will be used. For each
        level in bin_levels this determines how many absissas comprise the
        levels. Can be either 'gp' for Gauss-Patterson or 'cc' for
        Clenshaw-Curtis.

    Returns
    -------
    collect_components : int list
        Enumerations of each list in bin_levels.

    Example
    -------
    Consider a problem with 2 bins and we want the sum of the bins to equal 3.
    From the routine enumerate_bins_sum(2,3) we will get,
    
    bin_levels = [[2, 1], [1, 2]]

    Since each level consists of a specific number of nodes depending on the
    quadrature/interpolation scheme used, we want to enumerate through all of
    these since they are enumerated in the construction of Smolyak sparse grids.
    Using Gauss-Patterson absissas the command enumerate_upto_number_nodes(2,
    bin_levels,'gp') will return:
    
    [  [[1, 1], [2, 1], [3, 1]],
       [[1, 1], [1, 2], [1, 3]]  ]

    Notes
    -----
    The enumeration algorithm used here follows from a description in the book
    Sparse Grid Quadrature in High Dimensions with Applications in Finance and
    Insurance by Markus Holtz (Algorithm 3.3).
    """
       
    collect_components = []
    
    # Initialize
    p = 0
    s = [0] + [1]*(d-1)
    go = 1
    tensor_components = []

    # Outside loop
    while go == 1:
        # Obtain valid component of index expansion
        s[p] = s[p] + 1
        if s[p] > mi[p]:
            if p == (d-1):
                collect_components += [tensor_components]
                go = 0
            s[p] = 1
            p = p + 1
        else:
            p = 0
            # Valid component
            tensor_components += [list(s)]
                
    return collect_components[0]
