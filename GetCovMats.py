#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import reshape,eye,zeros,ones,concatenate
from CalcADelta import CalcADelta
from CalcB import CalcB
from CalcU import CalcU

def GetCovMats(D,Obs,Prior):
    
    M=D.nR * D.nt
    N=D.nR *(D.nt-1)
    
    DeltaA=CalcADelta(D.nR,D.nt)
    B=CalcB(D.nR,D.nt)

    stop=1
    
    Obs.Ch=Obs.sigh**2 * eye(M)
    Obs.Cw=Obs.sigw**2 * eye(M)
    Obs.CS=Obs.sigS**2 * eye(M)
    
    # calculate Jacobian of the dAdt term wrt h and w
    Obs.JAh=(B @ Obs.wv @ ones((1,M)))*DeltaA/(D.dt @ ones((1,M)))
    Obs.JAw=(DeltaA @ Obs.hv @ ones((1,M)) ) * B / (D.dt @ ones((1,M)))
    Obs.JA=concatenate((Obs.JAh,Obs.JAw),1)
    
    #now calculate the covariance of the dAdt term w.r.t w&h
    Chwtop=concatenate((Obs.Ch,zeros((M,M))),1)
    Chwbot=concatenate((zeros((M,M)),Obs.Cw),1 )
    Chw=concatenate((Chwtop,Chwbot),0 )
    
    Obs.CA=Obs.JA @ Chw @ Obs.JA.T
    
    #calculate covariance matrix of the dA term based on width & height errors
    U=CalcU(D)
    JdAh= U @ ( (ones((N,1)) @ Obs.wv.T) * DeltaA )
    JdAw= U @ ( (DeltaA @ Obs.hv @ ones((1,M)) ) * B )
    JdA=concatenate( (JdAh,JdAw),1  )
    
    Obs.CdA=JdA @ Chw @ JdA.T
    
    Obs.CSdAw=concatenate( (concatenate((Obs.CS,zeros((M,M)),zeros((M,M))),1),\
                            concatenate((zeros((M,M)),Obs.CdA,zeros((M,M))),1),\
                            concatenate((zeros((M,M)),zeros((M,M)),Obs.Cw),1)),0) 
    
    Prior.Cqf=0
    
    
    return Obs,Prior
    
    