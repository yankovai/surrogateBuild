import unittest
from Clenshaw_Curtis import cc_data_main
import numpy as np

class test_Clenshaw_Curtis(unittest.TestCase):
    """
    Tests to see if the Clenshaw-Curtis abscissas/barycentric weights actually
    interpolate as they should. 
    """
    
    def setUp(self):
        """
        Establish polynomial that is going to be interpolated.
        """

        self.cc_data = cc_data_main()
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
  
    def test_cc_interpolation(self):

        # Initialize
        cc_data = self.cc_data
        # Test points
        nx_test = 50
        x_test = np.linspace(-1.,1.,nx_test)
        f_test = self.f(x_test)

        # Calculate interpolant value using barycentric interpolation for
        # several levels of Clenshaw-Curtis abscissas. 
        for level in range(1,11):
            nknots = cc_data[level].nknots
            x_cc = cc_data[level].knots
            w_cc = cc_data[level].bweights
            f_x_cc = self.f(x_cc)
            
            Xtest = np.tile(x_test, [nknots,1])
            Xcc = np.tile(x_cc, [nx_test,1]).transpose()
            W = np.tile(w_cc, [nx_test,1]).transpose()
            F = np.tile(f_x_cc, [nx_test,1]).transpose()
            
            W_dX = W/(Xtest - Xcc + self.eps)
            interpvals = np.sum(W_dX*F,0)/np.sum(W_dX,0)
            err = np.linalg.norm(interpvals - f_test)
            if err < 1e-8:
                for i in range(nx_test):
                    self.assertAlmostEquals(interpvals[i],f_test[i],delta=1e-8)
            
if __name__ == '__main__':
    unittest.main()
