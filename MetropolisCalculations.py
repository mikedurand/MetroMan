#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:57:11 2020

@author: mtd
"""

from numpy import empty,mean,exp,putmask,log,any
from scipy.stats import lognorm
import time
from CalcDelta import CalcDelta
from CalcADelta import CalcADelta
from CalcB import CalcB
from logninvstat import logninvstat
from CalcLklhd import CalcLklhd


def MetropolisCalculations(Prior,D,Obs,jmp,C,R,DAll,AllObs,nOpt,DebugMode):
    [Delta,DeltaA,B,C,thetauA0,thetauna,thetaux1,thetauq,R]=InitializeMetropolis(D,C,Prior,R)
    
    if DebugMode:
        C.N=int(C.N/10)
        C.Nburn=int(C.Nburn/10)        
    
    jmp.stdA0=0.1*mean(thetauA0)
    jmp.stdna=0.01*mean(thetauna)
    jmp.stdx1=0.1*mean(thetaux1)
    
    # set target acceptance rates to 0.25 since all quantities are vectors (length D.nR)
    jmp.target1=0.25
    jmp.target2=0.25
    jmp.target3=0.25
    
    jmp.stdA0s=empty((C.N))
    jmp.stdnas=empty((C.N))
    jmp.stdx1s=empty((C.N))
    
    meanA0=Prior.meanA0
    covA0=Prior.stdA0/meanA0
    vA0=(covA0*meanA0)**2
    [muA0,sigmaA0]=logninvstat(meanA0,vA0)
    
    #%%
    meanna=Prior.meanna
    covna=Prior.stdna/meanna
    vna=(covna*Prior.meanna)**2
    [muna,sigmana] = logninvstat(meanna,vna)
    
    meanx1=Prior.meanx1
    covx1=Prior.stdx1/meanx1
    vx1=(covx1*Prior.meanx1)**2
    [mux1,sigmax1] = logninvstat(-Prior.meanx1,vx1)

    pu1=lognorm.pdf(thetauA0,sigmaA0,0,exp(muA0))
    pu2=lognorm.pdf(thetauna,sigmana,0,exp(muna))
    
    if nOpt<5:
        pu3=lognorm.pdf(-thetaux1,sigmax1,0,exp(mux1))
    elif nOpt==5:
        pu3=lognorm.pdf(thetaux1,sigmax1,0,exp(mux1))
    
    fu=CalcLklhd(Obs,AllObs,thetauA0,thetauna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt)
    
    C.n_a1=0
    C.n_a2=0
    C.n_a3=0
    
    C.Like=empty((C.N))
    C.LogLike=empty((C.N))
    
    #%%
    tic=time.process_time()
    for i in range(0,C.N):
        if i%1000==0:
            print("Iteration #", i+1, "/", C.N, ".")
        if i<C.N*.2 and i>0 and i%100==0:
            jmp.stdA0=mean(jmp.stdA0s[0:i-1] )/jmp.target1*(C.n_a1/i)
            jmp.stdna=mean(jmp.stdnas[0:i-1] )/jmp.target2*(C.n_a2/i)
            jmp.stdx1=mean(jmp.stdx1s[0:i-1] )/jmp.target3*(C.n_a3/i)

        jmp.stdA0s[i]=jmp.stdA0    
        jmp.stdnas[i]=jmp.stdna
        jmp.stdx1s[i]=jmp.stdx1
        
        #A0
        thetavA0=thetauA0+jmp.stdA0*R.z1[:,i]       
        thetavA0[thetavA0<jmp.A0min.reshape((D.nR,))]=putmask(thetavA0,thetavA0<jmp.A0min,jmp.A0min)
        
        pv1=lognorm.pdf(thetavA0,sigmaA0,0,exp(muA0))
        fv=CalcLklhd(Obs,AllObs,thetavA0,thetauna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt)
        
        MetRatio=exp(fv-fu)*exp(sum(log(pv1))-sum(log(pu1)))
        if MetRatio > R.u1[i]:
            C.n_a1=C.n_a1+1
            thetauA0=thetavA0; fu=fv;pu1=pv1 # update u->v
        C.thetaA0[:,i]=thetauA0.T
        
        #na
        thetavna=thetauna+jmp.stdna*R.z2[:,i]
        thetavna[thetavna<jmp.nmin]=putmask(thetavna,thetavna<jmp.nmin,jmp.nmin)
        
        pv2=lognorm.pdf(thetavna,sigmana,0,exp(muna))
        fv=CalcLklhd(Obs,AllObs,thetauA0,thetavna,thetaux1,D,Prior,Delta,DeltaA,B,thetauq,nOpt)
        
        MetRatio=exp(fv-fu)*exp(sum(log(pv2))-sum(log(pu2)))
        if MetRatio > R.u2[i]:
            C.n_a2=C.n_a2+1
            thetauna=thetavna; fu=fv; pu2=pv2;
        C.thetana[:,i]=thetauna.T
        
        #x1
        thetavx1=thetaux1+jmp.stdx1*R.z3[:,i]
        
        if nOpt<5:
            pv3=lognorm.pdf(-thetavx1,sigmax1,0,exp(mux1))
        elif nOpt==5:
            pv3=lognorm.pdf(thetavx1,sigmax1,0,exp(mux1))

        fv=CalcLklhd(Obs,AllObs,thetauA0,thetauna,thetavx1,D,Prior,Delta,DeltaA,B,thetauq,nOpt)
        
        if any(pv3==0):
            MetRatio=0
        else:
            MetRatio=exp(fv-fu)*exp(sum(log(pv3))-sum(log(pu3)))
        
        if MetRatio > R.u3[i]:
            C.n_a3=C.n_a3+1
            thetaux1=thetavx1; fu=fv; pu3=pv3;
        C.thetax1[:,i]=thetaux1.T
        
        C.Like[i]=exp(fu)
        C.LogLike[i]=fu
    
    toc=time.process_time(); print('McFLI MCMC Time: %.2fs' %(toc-tic))
    
    print('A0: Acceptance rate =',(C.n_a1/C.N*100), ' pct.')
    print('na: Acceptance rate =', (C.n_a2/C.N*100), ' pct.')
    print('x1 Acceptance rate =', (C.n_a3/C.N*100), ' pct.')
    
    #%%
    return C

def InitializeMetropolis(D,C,P,R):
    from numpy.random import seed,rand,randn
    
    
    Delta=CalcDelta(D.nR,D.nt,D.L)
    DeltaA=CalcADelta(D.nR,D.nt)
    B=CalcB(D.nR,D.nt)
    
    C.thetaA0=empty((D.nR,C.N))
    C.thetaA0[:,0]=P.meanA0
    thetauA0=C.thetaA0[:,0]
    
    C.thetana=empty((D.nR,C.N))    
    C.thetana[:,0]=P.meanna
    thetauna=C.thetana[:,0]
    
    C.thetax1=empty((D.nR,C.N))
    C.thetax1[:,0]=P.meanx1
    thetaux1=C.thetax1[:,0]
    
    thetauq=[]
    
    seed([R.Seed])
    
    R.z1=randn(D.nR,C.N)
    R.z2=randn(D.nR,C.N)
    R.z3=randn(D.nR,C.N)
    R.u1=rand(C.N,1)
    R.u2=rand(C.N,1)
    R.u3=rand(C.N,1)
    
    return Delta,DeltaA,B,C,thetauA0,thetauna,thetaux1,thetauq,R


