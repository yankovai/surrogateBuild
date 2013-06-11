import numpy as np

def number_knots(level):
    """
    Takes as input an array of different levels and returns the number of knots
    in each respective level. 
    """

    nknots = 2**(level-1) + 1
    where_equals2 = nknots == 2
    nknots[where_equals2] = 1
    return nknots

def barycentric_weights(mi):
    """
    """

    weights = np.ones(mi)
    weights[0] = .5
    weights[-1] = .5
    weights[1:-1:2] *= -1.
    
    return weights

def knots(mi):
    """
    """
    
    if mi > 1:
        j = np.array(range(1,mi+1))
        return np.cos(np.pi*(j-1.)/(mi-1.))
    else:
        return np.array([np.cos(.5*np.pi)])

class Clenshaw_Curtis:
    """
    Contains information about Clenshaw-Curtis abscissas.
    """

    def __init__(self,level):
        """
        level = the degree of abscissas, which define everything else.
        knots = Clenshaw-Curtis abscissas for level
        nknots = number of knots
        bweights = barycentric weights for knots
        """
        
        self.level = level
        self.nknots = number_knots(np.array([level]))[0]
        self.knots = knots(self.nknots)    
        self.bweights = barycentric_weights(self.nknots)

def cc_data_main(maxlevel=10):
    """
    """

    cc_data = {}
    cc_data.setdefault('nknots',number_knots)
    for level in range(1,maxlevel+1):
        cc_data.setdefault(level,Clenshaw_Curtis(level))

    return cc_data


        

