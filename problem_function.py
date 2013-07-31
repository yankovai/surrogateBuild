import pickle
import subprocess
import numpy as np
from initialize_surrogate import Hypercube
import os

class Problem_Function(Hypercube):
    """
    Contains the function for which a surrogate model is desired along with
    information about the input variables.
    """
    
    def __init__(self,dactive,quad_type):
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
        
        self.d = 25
        self.mu = pickle.load(open('mu.dat','rb'))
        self.std = pickle.load(open('std.dat','rb'))
        self.covmatrix = pickle.load(open('covmatrix.dat','rb'))
        
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

        # Write cross-section file
        id_num, new_xsec_file = self._write_xsec_file(x)
        
        # Execute parcs with new xsec file
        new_input_file = self._write_input_file(id_num)
        
        results = subprocess.check_output(['../parcs.x',new_input_file])   
        keff = self._read_output_file(id_num)

        os.chdir('../')
        subprocess.call(['rm','-rf',id_num])
        
        return keff

    def _read_output_file(self,id_num):
        """
        """
        output_name = id_num + '.out'

        output = open(output_name,'r')
        for line in output:
            if 'K-Effective:' in line:
                keff = float(line.split()[1])
                break
                
        output.close()

        return keff

    def _write_xsec_file(self,x):
        """
        """
        
        # make random string identifier
        np.random.seed()
        id_num = str(x[0])[-10] + str(np.random.randint(100000,999999))
        # make sure directory doesn't already exist
        while os.path.exists(id_num):
            np.random.seed()
            id_num = str(x[0])[-10] + str(np.random.randint(100000,999999))

        # create new directory
        subprocess.call(['mkdir',id_num])
        os.chdir(id_num)

        target_xsec_names = pickle.load(open('../target_xsec_names.dat','rb'))
        xsec_template = open('../parcs.xsec','r')
        new_xsec_file = 'parcs.xsec.' + id_num
        xsec_new = open(new_xsec_file,'w')

        for line in xsec_template:
            for xsec_name, xsec in zip(target_xsec_names,x):
                if xsec_name in line:
                    line = line.replace(xsec_name,'%12.6E' %(xsec)) 
            xsec_new.write(line)

        xsec_new.close()
        xsec_template.close()

        return id_num, new_xsec_file

    def _write_input_file(self,id_num):
        """
        """

        input_template = open('../parcs.inp','r')
        new_input_file = 'parcs.inp.' + id_num
        input_new = open(new_input_file,'w')
        new_xsec_file = 'parcs.xsec.' + id_num
              
        for line in input_template:
            if 'file ./parcs.xsec.tmp' in line:
                line = line.replace('parcs.xsec.tmp', new_xsec_file)
            if 'CASEID' in line:
                line = line.replace('tmi_minicore', id_num)
            input_new.write(line)

        input_template.close()
        input_new.close()

        return new_input_file
    
##P = Problem_Function(range(25),'cc')
##x = np.random.multivariate_normal(P.mu,P.covmatrix)
##P.evalf_unnormalized_x(x)










