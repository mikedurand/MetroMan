#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class Domain:
    def __init__(self):
        self.nR=[] #number of reaches
        self.xkm=[] #reach midpoint distance downstream [m]
        self.L=[]  #reach lengths, [m]
        self.nt=[] #number of overpasses
        self.t=[] #time, [days]
        self.dt=[] #time delta between successive overpasses, [seconds]
        

class Observations:
    def __init__(self,D):
        from numpy import empty
        self.h=empty(  (D.nR,D.nt) ) #water surface elevation (wse), [m]
        self.h0=empty( (D.nR,1)  ) #initial wse, [m]
        self.S=empty(  (D.nR,D.nt) ) #water surface slope, [-]
        self.w=empty(  (D.nR,D.nt) ) #river top width, [m]
        self.hv=empty( (D.nR*D.nt,1) ) #reshaped version of h
        self.Sv=empty( (D.nR*D.nt,1) ) #reshaped version of S
        self.wv=empty( (D.nR*D.nt,1) ) #reshaped version of w
        self.sigh=[] #wse uncertainty standard deviation [m]
        self.sigS=[] #slope uncertainty standard deviation [-]
        self.sigW=[] #width uncertainty standard deviation [m]
        self.dA=empty( (D.nR,D.nt) ) #cross-sectional area anomaly [m^2]
        self.dAv=empty( (D.nR*(D.nt-1)) ) #reshaped version of dA
        self.hmin=empty( (D.nR,1)  ) #minimum wse, [m]
        self.A0Shift=empty( (D.nR,1)  ) #difference in A0 between inversion window and all of timeseries
        self.Ch=empty( (D.nR*D.nt,D.nR*D.nt))
        self.Cw=empty( (D.nR*D.nt,D.nR*D.nt))
        self.CS=empty( (D.nR*D.nt,D.nR*D.nt))
        
        #have not yet added the Jacobians and additional covariance matrices
        #just ran out of time. these are all defined in GetCovMats
        
        self.Cqf=[] #covariance matrix of lateral inflows. 
                   #not currently used
        
class Chain:
    def __init__(self):
        self.N=[] #total number of markov chain iterations
        self.Nburn=[] #number of markov chain iterations in the "burn in" period
        self.thetaA0=[] #these should be initialized with D.nR x C.N... hope to fix in future
        self.thetana=[]
        self.thetax1=[]
        self.Like=[] #should be initialized with C.Nx1
        self.LogLike=[]

class RandomSeeds:
    def __init__(self):
        self.Seed=[] #seed for random number generator
        self.z1=[]  #these should be initialized with D.nR x C.N... hope to fix in future
        self.z2=[]
        self.z3=[]
        self.u1=[]
        self.u2=[]
        self.u3=[]
        
class Experiment:
    def __init__(self):
        from numpy import empty        
        self.tUse=empty( (1,2) ) #time indices to use for the inversion window, assuming 1 is first index (matlab style)
        self.nOpt=[]  #friction coefficient parameterization option
        self.tStep=[] #time step of temporal data [day]
        
class Prior:
    def  __init__(self,D):
        from numpy import empty           
        self.meanQbar=[] #mean of the prior estimate of mean annual flow [m^3/s]
        self.covQbar=[] #coefficient of variation of the prior estimate of mean annual flow [-]
        self.meanAllA0=empty( (D.nR,1) )
        self.stdAllA0=empty( (D.nR,1) )
        self.meanna=empty( (D.nR,1) )
        self.stdna=empty( (D.nR,1) )
        self.meanx1=empty( (D.nR,1) )    
        self.stdx1=empty( (D.nR,1) )    
        self.meanA0=empty( (D.nR,1) )
        self.stdA0=empty( (D.nR,1) )
        
    class Geomorph:
        def __init(self):
            self.Use=[]
            self.loga_hat=[] 
            self.loga_sigma=[]
            self.b_hat=[]
            self.b_sigma
            self.logA0_hat=[]
            self.logA0_sigma=[]
                
    class Lats:
        def __init__(self,D):
            from numpy import empty   
            self.q=empty( (D.nR,D.nt-1) )  #lateral inflows [m^2/s]
            self.qv=empty( (D.nR* (D.nt-1),1 )  )
    class AllLats:
        def __init__(self,D):
            from numpy import empty   
            self.q=empty( (D.nR,D.nt-1) )  #lateral inflows [m^2/s]
            self.qv=empty( (D.nR* (D.nt-1),1 )  )            
                
class Truth:
    def __init__(self,D):
        from numpy import empty 
        self.A0=[] #cross-sectional area for first time index [m^2]
        self.q=[] #lateral inflow [m^2/s]
        self.n=[] #friction coefficient []
        self.Q=empty( (D.nR,D.nt)  ) #discharge [m^3/s]
        self.dA=empty( (D.nR,D.nt)  ) #cross-sectional area change [m^3/s]
        self.W=empty( (D.nR,D.nt)  ) #width [m]
        self.h=empty( (D.nR,D.nt)  ) #wse [m]

class Jump:
    def __init__(self):
        self.stdA0burn=[]
        self.stdA0sim=[]
        self.A0min=[]
        self.stdn=[]
        self.nmin=[]
        self.stdq=[]
        self.qmin=[]
        self.target1=[]
        self.target2=[]
        self.target3=[]
        self.stdA0s=[]
class Estimates:
    def __init__(self,D,DAll):
        from numpy import empty
        self.A0hat=empty((D.nR,1))
        self.CA0=empty((D.nR,D.nR))
        #not including remainder of those for now...
        self.nhat=empty((D.nR,D.nt))
        self.nhatAll=empty((DAll.nR,DAll.nt))
        self.AllQ=empty((DAll.nR,DAll.nt))

class ErrorStats:
    def __init__(self):
        self.RMSE=[]
        self.nRMSE=[]
        