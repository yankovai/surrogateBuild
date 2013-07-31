import multiprocessing as mp
import numpy as np
import os

def f(x):
    pid = os.getppid()
    print 'I am process # %d and I have integer value %d' %(pid,x)
    return x
    

if __name__ == '__main__':
    ndata = 10
    data = np.random.randint(150,275,ndata)

    po = mp.Pool()
    print 'Using a total of %d processes' %(mp.cpu_count())
    fvals = po.map(f,data)
    po.close()
    po.join()

    rio = (data == np.array(fvals)).all()
    print 'Are the results in order? ', rio

