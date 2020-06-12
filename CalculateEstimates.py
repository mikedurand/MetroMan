#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:52:52 2020

@author: mtd
"""
from numpy import mean,cov,sqrt,diagonal
from MetroManVariables import Estimates 
from calcnhat import calcnhat

def CalculateEstimates(C,D,Obs,Prior,DAll,AllObs,nOpt):
    
    #1) estimates on the chain A0, n, q: means & covariances
    E=Estimates(D,DAll)
    E.A0hat=mean(C.thetaA0[:,C.Nburn+1:C.N],1)    
    E.CA0=cov(C.thetaA0[:,C.Nburn+1:C.N])
    E.stdA0Post=sqrt(diagonal(E.CA0)) 
    
    E.nahat=mean(C.thetana[:,C.Nburn+1:C.N],1)    
    E.Cna=cov(C.thetana[:,C.Nburn+1:C.N])    
    E.stdnaPost=sqrt(diagonal(E.Cna)) 

    E.x1hat=mean(C.thetax1[:,C.Nburn+1:C.N],1)    
    E.Cx1=cov(C.thetax1[:,C.Nburn+1:C.N])    
    E.stdx1Post=sqrt(diagonal(E.Cx1))
    
    for i in range(0,D.nR):
        E.nhat[i,:]=calcnhat(Obs.w[i,:], Obs.h[i,:], Obs.hmin[i], E.A0hat[i]+Obs.dA[i,:], E.x1hat[i], E.nahat[i], nOpt)
        E.nhatAll[i,:]=calcnhat(AllObs.w[i,:], AllObs.h[i,:], AllObs.hmin[i], E.A0hat[i]+AllObs.dA[i,:]-AllObs.A0Shift[i], E.x1hat[i], E.nahat[i], nOpt)
    
    #2)calculate the Q chain, and estimate mean and std
    #... not translated for now
    #5)all estimates
    for i in range(0,D.nR):
        E.AllQ[i,:]=1/E.nhatAll[i,:]*(E.A0hat[i]+AllObs.dA[i,:]-AllObs.A0Shift[i])**(5/3) * AllObs.w[i,:]**(-2/3) * AllObs.S[i,:]**0.5
    
    stop=1
    
    return E,C