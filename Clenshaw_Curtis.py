import numpy as np

def number_knots(level):
    """
    """

    if level == 1:
        return 1
    else:
        return 2**(level-1) + 1


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
        self.nknots = number_knots(level)
        self.knots = knots(self.nknots)    
        self.bweights = barycentric_weights(self.nknots)

def cc_data_main(maxlevel=10):
    """
    """

    cc_data = {}
    for level in range(1,maxlevel+1):
        cc_data.setdefault(level,Clenshaw_Curtis(level))

    return cc_data

