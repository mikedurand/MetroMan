from numpy import ones,empty,maximum,sqrt

def ManningCalcs(A0,n,S,dA,wobs):

    # should move this to its own function eventually, but would then need to
    # redo all calling functions... ought to set this up as a trapezoidal
    # difference, eventually. This is mathematically OK but otherwise a bity
    # spotty.
   
    [nR,nt]=dA.shape
    A=dA+(A0.T*ones([1,nt]))
    Q=empty([nR,nt])
    
    for j in range(0,nt):  # loop over time
        Sj=maximum(S[:,j],9e-8)        
        Q[:,j]=(1/n[0])*(A[:,j]**(5./3.))*(wobs[:,j]**(-2./3.))*sqrt(Sj);

    return Q
