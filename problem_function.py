from scipy.integrate import odeint
import numpy as np
from initialize_surrogate import Hypercube

class Problem_Function(Hypercube):
    """
    Contains the function for which a surrogate model is desired along with
    information about the input variables.
    """
    
    def __init__(self,dactive, quad_type):
        """
        Parameters
        ----------
        d : int
            Number of dimensions in the function. Put another way, the number of
            inputs that go into the function.
        mu : float array
            Mean of all input variables.
        std : float array
            Standard deviation of all input variables.
        corrmatrix : float array
            Correlation matrix for the input variables.
        covmatrix : float array
            Covariance matrix for the input variables.
        dactive : int list
            A list of dimensions [1,2,3,...,k] that are not anchored, meaning
            their values are free to change. All other dimensions are fixed at
            the anchor value.
        """
        
        self.d = 22
        self.mu = np.array([0.0124,     # 0  lambda 1
                            0.0305,     # 1  lambda 2
                            0.1110,     # 2  lambda 3
                            0.3010,     # 3  lambda 4
                            1.1400,     # 4  lambda 5
                            3.0100,     # 5  lambda 6
                            0.00009,    # 6  beta 1
                            0.000853,   # 7  beta 2 
                            0.0007,     # 8  beta 3
                            0.0014,     # 9  beta 4
                            0.0006,     # 10 beta 5
                            0.00055,    # 11 beta 6
                            4e-7,       # 12 Lambda
                            2.5e6,      # 13 Ah
                            1168.,      # 14 M_coolant
                            9675.,      # 15 M_fuel
                            1200.,      # 16 c_p coolant
                            500.,       # 17 c_p fuel
                            7.5,        # 18 velocity
                            6.87e-6,    # 19 alpha fuel 
                            1.23e-6,    # 20 alpha coolant
                            0.0004193]) # 21 parameter in external reactivity
  
        self.std = .05*self.mu
        # Correlation matrix for inputs
        self.corrmatrix = np.diag(np.ones(22))
        # Build covariance matrix for the inputs
        s = self.std
        covmatrix = np.zeros([22,22])
        for i in range(0,self.d):
            for j in range(0,self.d):
                covmatrix[i,j] = self.corrmatrix[i,j]*s[i]*s[j]
        self.covmatrix = covmatrix
        self.dactive = list(dactive)
        self.quad_type = quad_type
        
        Hypercube.__init__(self)
        self._dactive_covariance()
        self._dactive_mean()
        
    def evalf_unnormalized_x(self,x):
        """
        Parameters
        ----------
        x : float array
            The point at which the function is to be evaluated.
            
        Returns
        -------
        f(x) : float
            Function value at x.
        """

        # Initialize
        lambda_k = x[0:6]
        beta_k = x[6:12]
        Lmbda = x[12]
        Ah = x[13]
        M_c = x[14]
        M_f = x[15]
        c_pc = x[16]
        c_pf = x[17]
        v = x[18]
        Tin = 627.15
        L = 0.91
         
        # Initial Conditions
        P0 = 2.1e9 # Watts
        C0 = beta_k*P0/(lambda_k*Lmbda)
        Tc0 = Tin + P0*L/(M_c*c_pc*v)
        Tf0 = Tc0 + P0/Ah
        y0 = [P0] + list(C0) + [Tc0] + [Tf0]
        args = tuple(x) + (Tc0,) + (Tf0,) + (Tin,) + (L,)
        # Solve
        t = np.linspace(0.,30.,500)  # changed from (0.,50.,1000)
        soln = odeint(self.__kth_system,y0,t,args)

        # Return relative maximum power
        return max(soln[:,0])/P0

    def __kth_system(self,y,t,*args):
        args = np.array(args)
        P = y[0]
        C = y[1:7]
        Tc = y[7]
        Tf = y[8]
        if t < 20:
            rho_e = (t/20.)*args[21]
        else:
            rho_e = 0.
        rho = rho_e - args[19]*(Tf - args[23]) + args[20]*(Tc - args[22])
        f0 = (rho - np.sum(args[6:12]))*P/args[12] + np.sum(args[0:6]*C)
        f1 = args[6]*P/args[12] - args[0]*C[0]
        f2 = args[7]*P/args[12] - args[1]*C[1]
        f3 = args[8]*P/args[12] - args[2]*C[2]
        f4 = args[9]*P/args[12] - args[3]*C[3]
        f5 = args[10]*P/args[12] - args[4]*C[4]
        f6 = args[11]*P/args[12] - args[5]*C[5]
        f7 = args[13]*(Tf-Tc)/(args[14]*args[16]) - args[18]*(Tc - args[24])/args[25]
        f8 = (P + args[13]*(Tc - Tf))/(args[15]*args[17])
        return [f0, f1, f2, f3, f4, f5, f6, f7, f8]



