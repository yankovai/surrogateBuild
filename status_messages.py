class Status_Message_Surrogate:
    """
    Print status messages during the build of the reduced-order model for the
    objective function.
    """

    def __init__(self):
        """
        Returns
        -------
        out : string
            Tells user that the routines for construction of the reduced-order
            model have begun.
        """

        print ' '
        print 'Begin Construction of Reduced-Order Model'.center(80)
        print ''.center(80,'-')
        print ' '

    def build_order(self,norder):
        """
        Parameters
        ----------
        norder : int
            The order of components being constructed in the HDMR expansion.

        Returns
        -------
        out : string
            Tells the user the order of HDMR components currently being
            executed.
        """

        out = 'Building components of order: %d' %(norder)
        print out.center(80)

    def build_hdmr_component(self,dactive):
        """
        Parameters
        ----------
        dactive: tuple
            The active dimensions for which a Smolyak sparse grid interpolant
            is currently being built.

        Returns
        -------
        out : string
            Tells the user which subset of total dimensions are currently being
            investigated by the algorithm. 
        """

        print ' '
        print ' Active Dimensions:', dactive
        print ' Build Interpolant for HDMR component...'
        
    def order_info(self,mean,var):
        """
        Parameters
        ----------
        mean : float
            The mean of the reduced-order model after calculating all the
            contributions of an HDMR order.
        var : float
            The variance of the reduced-order model after calculating all the
            contributions of an HDMR order.

        Returns
        -------
        out : string
            Tells the user what the mean and variance are after calculating all
            contributions from available HDMR orders.
        """
        
        print ' '
        print ''.center(80,'-')
        print 'Statistics for Reduced-Order Model'.center(80) 
        out = 'Mean %11.5E' %(mean)
        print out.center(80)
        out = 'Variance %11.5E' %(var)
        print out.center(80)
        print ''.center(80,'-')
        print ' '
    
class Status_Message_SparseGrid:
    """
    Prints status messages during construction of Smolyak interpolants for the
    objective function. 
    """
    
    def __init__(self):
        """
        Returns
        -------
        output : string
            Initiatalizing statement printed when class is loaded.
        """

        print ' '
        print '  Level     Mean     Variance   FEvals    Max HS    Converged  '
        print ' ------- ---------- ---------- -------- ---------- ----------- '
        
    def level_info(self,smolyak_level,mean,var,fvals,maxhs,conv):
        """
        Parameters
        ----------
        smolyak_level : int
            Current smolyak level of surrogate build.
        mean : float
            Mean of surrogate at current smolyak level.
        var : float
            Variance of surrogate at current smolyak level.
        fvals : int
            Total number of function evaluations required so far.
        maxhs : float
            Maximum hierarchical surplus at current smolyak level
        conv : boolean
            States whether surrogate is converged yet.

        Returns
        -------
        output : string
            Regurgitates the input parameters in a visually friendly way.
        """

        print '   %2d     %8.2E   %8.2E    %4d    %8.2E     %5s' \
              %(smolyak_level,mean,var,fvals,maxhs,str(conv)) 
