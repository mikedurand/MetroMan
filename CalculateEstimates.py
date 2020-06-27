#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:52:52 2020

@author: mtd
"""
from numpy import mean,cov,sqrt,diagonal,empty,NaN,ones,std,diag,log,corrcoef
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
    
    #2) calculate the Q chain, and estimate mean and std
    nhat=empty([D.nR,D.nt])
    nhat[:]=NaN
    nhatAll=empty([D.nR,DAll.nt])
    nhatAll[:]=NaN
    C.thetaQ=empty([C.N,D.nR,D.nt])
    C.thetaQ[:]=NaN
    C.thetaAllQ=empty([C.N,D.nR,DAll.nt])
    C.thetaAllQ[:]=NaN
    for i in range(0,C.N):
        for r in range(0,D.nR):
            nhat[r,:]=calcnhat(Obs.w[r,:], Obs.h[r,:], AllObs.hmin[r], \
                               E.A0hat[r]+Obs.dA[r,:], \
                               C.thetax1[r,i],C.thetana[r,i],nOpt)
            nhatAll[r,:]=calcnhat(AllObs.w[r,:], AllObs.h[r,:], AllObs.hmin[r], \
                               E.A0hat[r]+AllObs.dA[r,:], \
                               C.thetax1[r,i],C.thetana[r,i],nOpt)
        C.thetaQ[i,:,:]=1/nhat*(C.thetaA0[:,i].reshape(D.nR,1) @ ones([1,D.nt])\
                                  +Obs.dA)**(5/3) * Obs.w**(-2/3) * sqrt(Obs.S)
        C.thetaAllQ[i,:,:]=1/nhatAll*((C.thetaA0[:,i].reshape(D.nR,1)-AllObs.A0Shift.reshape(D.nR,1)) \
                                        @ ones([1,DAll.nt]) + AllObs.dA )**(5/3) * AllObs.w**(-2/3) \
                           *sqrt(AllObs.S)
        
    E.QhatPost=mean(C.thetaQ[C.Nburn:,:,:],0)
    E.QstdPost=std(C.thetaQ[C.Nburn:,:,:],0)
    
    #3) calculate Q prior estimate
    for r in range(0,D.nR):
        nhat[r,:]=calcnhat(Obs.w[r,:],Obs.h[r,:],AllObs.hmin[r], \
                           Prior.meanA0[r]*ones([1,D.nt])+Obs.dA[r,:], \
                           Prior.meanx1[r],Prior.meanna[r],nOpt); 

    E.QhatPrior=1/nhat * (Prior.meanA0.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)**(5/3) \
                *Obs.w**(-2/3)*sqrt(Obs.S);
    
    nhat=empty([D.nR,DAll.nt])
    nhat[:]=NaN
    for r in range(0,D.nR):
        nhat[r,:]=calcnhat(AllObs.w[r,:],AllObs.h[r,:],AllObs.hmin[r], \
                           Prior.meanA0[r]-AllObs.A0Shift[r]+AllObs.dA[r,:], \
                           Prior.meanx1[r],Prior.meanna[r],nOpt);                

    E.QhatAllPrior=1/nhat*((Prior.meanA0-AllObs.A0Shift).reshape(D.nR,1)@ones([1,DAll.nt])+AllObs.dA)**(5/3) \
                   *AllObs.w**(-2/3)*sqrt(AllObs.S);
    
    
    #4) discharge error budget: all done for Q(nr x nt)  
    #4.1) uncertainty estimate of the dA term
    Obs.sigdAv=sqrt(diag(Obs.CdA))
    Obs.sigdA=Obs.sigdAv.reshape(D.nt,D.nR).T
    #4.2) uncertainty (variance) of the Manning terms
    E.QhatUnc_na=(E.stdnaPost/Prior.meanna)**2
    E.QhatUnc_x1=(log(Obs.w/(E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)) * \
                    (E.stdx1Post.reshape(D.nR,1)@ones([1,D.nt])))**2;

    E.QhatUnc_w=(2/3*Obs.sigw/Obs.w)**2
    E.QhatUnc_S=(1/2*Obs.sigS/Obs.S)**2
    E.QhatUnc_A0=((5/3-Prior.meanx1.reshape(D.nR,1)@ones([1,D.nt])) * \
                 (E.stdA0Post.reshape(D.nR,1)@ones([1,D.nt])) / \
                 (E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA))**2
    E.QhatUnc_dA=((5/3-Prior.meanx1.reshape(D.nR,1)@ones([1,D.nt])) * \
                 Obs.sigdA / \
                 (E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA))**2
    
    #4.3) estimate correlation coefficient between A0 & na, A0 & x1, na & x1
    E.rho_A0na=empty([D.nR,1])
    E.rho_A0na[:]=NaN
    E.rho_A0x1=empty([D.nR,1])
    E.rho_A0x1[:]=NaN
    E.rho_nax1=empty([D.nR,1])
    E.rho_nax1[:]=NaN
    for i in range(0,D.nR):
        R_A0na=corrcoef(C.thetaA0[i,:], C.thetana[i,:])
        E.rho_A0na[i,0]=R_A0na[0,1]
        R_A0x1=corrcoef(C.thetaA0[i,:], C.thetax1[i,:])
        E.rho_A0x1[i,0]=R_A0x1[0,1]
        R_nax1=corrcoef(C.thetana[i,:], C.thetax1[i,:])
        E.rho_nax1[i,0]=R_nax1[0,1]

    #4.4) estimate uncertainty of Manning's Q
    #4.4.1) estimate uncertainty in Q due to cross-correlation of A0 & na, na & x1, x1 & A0
    E.QhatUnc_A0na=-2*(E.rho_A0na.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (5/3-Prior.meanx1.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdnaPost.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdA0Post.reshape(D.nR,1)@ones([1,D.nt])) / \
                      (Prior.meanna.reshape(D.nR,1)@ones([1,D.nt])) / \
                      (E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)
              
    E.QhatUnc_nax1=-2*(E.rho_nax1.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdx1Post.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdnaPost.reshape(D.nR,1)@ones([1,D.nt])) * \
                      log(Obs.w/(E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)) / \
                      (Prior.meanna.reshape(D.nR,1)@ones([1,D.nt]))
                  
    E.QhatUnc_A0x1=2*(E.rho_A0x1.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdA0Post.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (E.stdx1Post.reshape(D.nR,1)@ones([1,D.nt])) * \
                      (5/3-Prior.meanx1.reshape(D.nR,1)@ones([1,D.nt])) * \
                      log(Obs.w/(E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)) / \
                      (E.A0hat.reshape(D.nR,1)@ones([1,D.nt])+Obs.dA)
    
    #4.4.2) estimate total Q uncertinty
    E.QhatUnc_Hat=sqrt( E.QhatUnc_na+mean(E.QhatUnc_x1,1)+mean(E.QhatUnc_w,1)+ \
                        mean(E.QhatUnc_A0,1)+mean(E.QhatUnc_dA,1)+mean(E.QhatUnc_S,1)+ \
                        mean(E.QhatUnc_A0na,1)+mean(E.QhatUnc_nax1,1)+mean(E.QhatUnc_A0x1,1))
    
    #4.4.2) estimate total Q uncertinty wrt time
    E.QhatUnc_HatAll=sqrt( E.QhatUnc_na.reshape(D.nR,1)@ones([1,D.nt])+E.QhatUnc_x1+E.QhatUnc_w+ \
                           E.QhatUnc_A0+E.QhatUnc_dA+E.QhatUnc_S+ \
                           E.QhatUnc_A0na+E.QhatUnc_nax1+E.QhatUnc_A0x1)
    
    #4.4.3) discharge error budget
    E.QerrVarSum=empty([D.nR,6])
    E.QerrVarSum[:]=NaN
    E.QerrVarSum[:,0]=E.QhatUnc_na;
    E.QerrVarSum[:,1]=mean(E.QhatUnc_x1,1);
    E.QerrVarSum[:,2]=mean(E.QhatUnc_A0,1);
    E.QerrVarSum[:,3]=mean(E.QhatUnc_dA,1);
    E.QerrVarSum[:,4]=mean(E.QhatUnc_w,1);
    E.QerrVarSum[:,5]=mean(E.QhatUnc_S,1);

    #5)all estimates
    for i in range(0,D.nR):
        E.AllQ[i,:]=1/E.nhatAll[i,:]*(E.A0hat[i]+AllObs.dA[i,:]-AllObs.A0Shift[i])**(5/3) * AllObs.w[i,:]**(-2/3) * AllObs.S[i,:]**0.5
    
    
    return E,C