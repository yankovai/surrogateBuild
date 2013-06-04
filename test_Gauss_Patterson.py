import unittest
from Gauss_Patterson import gp_data_main
import numpy as np

class test_Gauss_Patterson(unittest.TestCase):
    """
    Tests to see if the Clenshaw-Curtis abscissas/barycentric weights actually
    interpolate as they should. 
    """
    
    def setUp(self):
        """
        Establish polynomial that is going to be interpolated.
        """

        self.gp_data = gp_data_main()
        self.eps = np.finfo(np.float64).eps        
        self.f_powers = np.random.random_integers(0,10,10)
        self.f_coeffs = np.random.randn(10) 

    def f(self,x):
        """
        Polynomial where powers and coefficients are chosen randomly.
        """

        n = len(x)
        fx = np.zeros(n)
        for i in range(0,n):
            fx[i] = np.sum(self.f_coeffs*x[i]**self.f_powers)
            
        return fx
  
    def test_gp_interpolation(self):

        # Initialize
        gp_data = self.gp_data
        # Test points
        nx_test = 50
        x_test = np.linspace(-1.,1.,nx_test)
        f_test = self.f(x_test)

        # Calculate interpolant value using barycentric interpolation for
        # several levels of Clenshaw-Curtis abscissas. 
        for level in range(1,6):
            nknots = gp_data[level].nknots
            x_gp = gp_data[level].knots
            w_gp = gp_data[level].bweights
            f_x_gp = self.f(x_gp)
            
            Xtest = np.tile(x_test, [nknots,1])
            Xgp = np.tile(x_gp, [nx_test,1]).transpose()
            W = np.tile(w_gp, [nx_test,1]).transpose()
            F = np.tile(f_x_gp, [nx_test,1]).transpose()
            
            W_dX = W/(Xtest - Xgp + self.eps)
            interpvals = np.sum(W_dX*F,0)/np.sum(W_dX,0)
            err = np.linalg.norm(interpvals - f_test)
            if err < 1e-8:
                for i in range(nx_test):
                    self.assertAlmostEquals(interpvals[i],f_test[i],delta=1e-8)
            
if __name__ == '__main__':
    unittest.main()
