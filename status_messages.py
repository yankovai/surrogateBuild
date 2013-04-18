
class Print_Status_Message:
    """
    
    """
    
    def __init__(self):
        """
        Returns
        -------
        output : string
            Initiatalizing statement printed when class is loaded.
        """
        print ' '
        print '---------------------------------------------------------------'
        print '--                Initializing Interpolation                 --'
        print '---------------------------------------------------------------'
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

        
        

    
    
