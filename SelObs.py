#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import reshape,diff,ones,empty,arange
from MetroManVariables import Observations,Domain,Truth

def SelObs(DAll,Obs,Exp,AllTruth,AllLats): 
    
    AllObs=Obs

    #%% set up domain for the inversion window
    D=Domain()
    D.nR=DAll.nR
    D.xkm=DAll.xkm
    D.L=DAll.L
    
    iEstBounds=arange(2)
    
    for i in range(0,DAll.nt):
        if DAll.t[0,i]==float(Exp.tUse[0]):
            iEstBounds[0]=int(i)
        if DAll.t[0,i]==float(Exp.tUse[1]):
            iEstBounds[1]=int(i)
        
    Exp.iEst=arange(iEstBounds[0],iEstBounds[1]+1)
    
    D.t=DAll.t[:,Exp.iEst]         
    D.nt=D.t.shape[1]
        
    D.dt=reshape(diff(D.t).T*86400. * ones((1,D.nR)),(D.nR*(D.nt-1),1))    
    
    #%% select observations
    Obs=Observations(D)
    
    for i in range(0,DAll.nR):
        Obs.h[i,:]=AllObs.h[i,Exp.iEst]
    for i in range(0,DAll.nR):
        Obs.w[i,:]=AllObs.w[i,Exp.iEst]
    for i in range(0,DAll.nR):
        Obs.S[i,:]=AllObs.S[i,Exp.iEst]        
    
    Obs.hv=reshape(Obs.h, (D.nR*D.nt,1) )
    Obs.Sv=reshape(Obs.S, (D.nR*D.nt,1) )
    Obs.wv=reshape(Obs.w, (D.nR*D.nt,1) )
    
    Obs.sigh=AllObs.sigh
    Obs.sigw=AllObs.sigw
    Obs.sigS=AllObs.sigS
       
    #%%select truth
    Tru=Truth(D)
    Tru.A0=AllTruth.A0
    Tru.n=AllTruth.n
    Tru.q=AllTruth.q
    for i in range(0,DAll.nR):
        Tru.Q[i,:]=AllTruth.Q[i,Exp.iEst]    
    
    #%%select lateral inflows    
    q=empty( (D.nR,D.nt-1) )
    for i in range(0,DAll.nR):
        q[i,:]=AllLats.q[i,Exp.iEst[0:-1] ]    
 
    
    
    return D,Obs,AllObs,DAll,Tru,q
    
