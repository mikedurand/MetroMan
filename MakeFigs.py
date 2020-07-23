#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:58:17 2020

@author: mtd
"""

from numpy import *
import matplotlib.pyplot as plt
from logninvstat import logninvstat
from scipy.stats import lognorm, norm
import matplotlib.axes as axes
from datetime import datetime

def MakeFigs(D,Truth,Prior,C,E,Err,AllTruth,DAll,AllObs):
    
    plt.figure(1)
    plt.subplot(311)
    plt.plot(C.thetaA0.T)
    plt.title('Baseflow cross-sectional area, m^2')
    plt.subplot(312)
    plt.plot(C.thetana.T)
    plt.title('Roughness coefficient power law coefficient')
    plt.subplot(313)
    plt.plot(C.thetax1.T)
    plt.title('Roughness coefficient power law exponent')
    plt.show()
    
    plt.figure(2)
    plt.hist(C.thetana[0,C.Nburn+1:C.N],bins=100)
    plt.title('Reach 0, Mean = %.4f' %(mean(C.thetana[0,C.Nburn+1:C.N])) )
    plt.show()
    
    plt.figure(3)
    plt.plot(DAll.t.T,E.AllQ[0,:].reshape((DAll.nt,1)),label='estimate' ) 
    plt.plot(DAll.t.T,AllTruth.Q[0,:].reshape((DAll.nt,1)),label='true' )
    plt.legend()
    plt.xlabel('time,days')
    plt.ylabel('discharge m^3/s')
    plt.show()
    
    plt.figure(4)
    h1=plt.errorbar(D.xkm/1000,E.A0hat,E.stdA0Post, linewidth=2.0,color='r',label='Estimate')
    hp=plt.errorbar(D.xkm/1000,Prior.meanA0,Prior.stdA0,linewidth=2,color='g',label='Prior')
    h2=plt.plot(D.xkm/1000,Truth.A0,linewidth=2,color='b',label='True')
    plt.xlabel('Flow distance, km')
    plt.ylabel('Cross-sectional area, m^2')
    plt.legend()
    plt.show()
    
    plt.figure(5)
    meanA0=Prior.meanA0
    covA0=Prior.stdA0/meanA0
    vA0=(covA0*meanA0)**2
    [muA0,sigmaA0] = logninvstat(meanA0,vA0)
    
    for i in range(0,D.nR):    
        plt.subplot(1,D.nR,i+1)
        x=range(0,int(max(C.thetaA0[i,:]))+1)
        y=lognorm.pdf(x,sigmaA0[i],0,exp(muA0[i])) 
        plt.plot(x,y/max(y)*C.N/20,'r--',lineWidth=2)
        plt.hist(C.thetaA0[i,C.Nburn+1:-1],50)
        plt.axvline(x=Truth.A0[i],linewidth=2, color='g')
        plt.axvline(x=E.A0hat[i],linewidth=2, color='k',linestyle='dashed')
        plt.title('Reach' + str(i+1))
        plt.xlabel('A_0, m^2')
        plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(6)
    if Truth.n is None:
        Truth.n = empty(size(Truth.A0))
        Truth.n[:] = nan

    for i in range(0,D.nR):
        plt.subplot(1,D.nR,i+1)
        x=linspace(0,max(C.thetana[i,:]),100)
        y=norm.pdf(x,Prior.meanna[i],Prior.stdna[i])
        plt.plot(x,y/max(y)*C.N/20,'r--',lineWidth=2)

        plt.hist(C.thetana[i,C.Nburn+1:-1],50)
        #plt.axvline(x=Truth.n[i],linewidth=2, color='g')
        plt.axvline(x=E.nahat[i],linewidth=2, color='k',linestyle='dashed')
        plt.title('Reach ' + str(i+1))
        plt.xlabel('n0, [-]')
        plt.ylabel('Frequency')  
    plt.show()
    
    plt.figure(7)
    for i in range(0,D.nR):
        plt.subplot(1,D.nR,i+1)
        x=linspace(0,max(C.thetax1[i,:]),100)
        y=norm.pdf(x,Prior.meanx1[i],Prior.stdx1[i])
        plt.plot(x,y/max(y)*C.N/20,'r--',lineWidth=2)
    
        plt.hist(C.thetax1[i,C.Nburn+1:-1],50) 
        plt.axvline(x=E.x1hat[i],linewidth=2, color='k',linestyle='dashed') 
        plt.title('Reach ' + str(i+1))
        plt.xlabel('x1, [-]')
        plt.ylabel('Frequency')     
    plt.show()
    
    Qbar=squeeze(mean(mean(C.thetaAllQ[:,:,:],1 ),1))
    plt.figure(8)
    plt.plot(Qbar)
    plt.axhline(y=mean(mean(Truth.Q)), color='r', linewidth=2)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Discharge, m^3/s')
    plt.show()
    
    plt.figure(9)
    plt.plot(D.t.reshape(size(D.t)),Truth.Q.T, color='b',lineWidth=2,label='True')
    plt.plot(D.t.reshape(size(D.t)),E.QhatPostf.T,color='r',lineWidth=2,label='MetroMan')
    plt.xlabel('Time, days')
    plt.ylabel('Discharge, m^3/s')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels)
    plt.show()
    
    plt.figure(10)
    plt.plot(range(0,D.nR),Err.QRelErrPrior,'.-',label='Prior')
    plt.plot(range(0,D.nR),Err.QRelErrPost,'.-',label='Posterior')
    plt.xlabel('Reach')
    plt.ylabel('Relative error')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure(11)
    plt.plot(Qbar,C.LogLike,'o', mfc='none')
    plt.xlabel('Average discharge, m^3/s')
    plt.ylabel('Log of likelihood')
    plt.show()
    
    plt.figure(12)
    plt.plot(DAll.t.reshape(DAll.nt),mean(AllTruth.Q,0),lineWidth=2,color='b',label='True')
    plt.plot(DAll.t.reshape(DAll.nt),mean(E.AllQ,0),lineWidth=2,color='r',label='Estimate')
    plt.plot(DAll.t.reshape(DAll.nt),mean(E.QhatAllPrior,0),lineWidth=2,color='y',label='Prior')
    plt.rcParams['font.size'] = 14
    plt.ylabel('Discharge, m^3/s')
    plt.legend()
    #if DAll.t[0] > date.toordinal(date(1900,0,0)):
    plt.show()
    
    r=1
    plt.figure(13)
    plt.subplot(1,2,1)
    plt.loglog(E.AllQ[r-1,:].T,AllObs.w[r-1,:].T,'+')
    plt.xlabel('Estimated Discharge, m^3/s')
    plt.ylabel('Width, m')
    plt.title('AHG for Reach #'+ str(r))

    plt.subplot(1,2,2)
    plt.plot(AllObs.h[r-1,:].T,AllObs.w[r-1,:].T,'+')
    plt.xlabel('Height, m')
    plt.ylabel('Width, m')
    plt.title('Stage-area for Reach #'+ str(r))
    plt.show()
                 
    #should make this a variable to be passed in... and should get the "true" slopes added to input file 
    iPos=where(AllObs.S > 1E-5)
    nTrue=empty([DAll.nR,DAll.nt])
    nTrue[:] = nan             
    ATrue=AllTruth.A0.T.reshape(D.nR,1)@ones([1,DAll.nt])+AllTruth.dA
    nTrue[iPos]=1/AllTruth.Q[iPos]*ATrue[iPos]**(5/3)*AllTruth.W[iPos]**(-2/3)*(AllObs.S[iPos])**.5

    plt.figure(14)
    r=range(0,DAll.nR)
    plt.subplot(1,2,1)
    plt.plot(E.AllQ[r,:].T,E.nhatAll[r,:].T,'o')
    plt.rcParams['font.size'] = 14
    plt.xlabel('Estimated Discharge, m^3/s')
    plt.ylabel('Estimated n, [-]')
    plt.subplot(1,2,2)
    plt.plot(AllTruth.Q[r,:].T,nTrue[r,:].T,'o')
    plt.xlabel('True Discharge, m^3/s')
    plt.ylabel('"True" n, [-]')
    plt.show()
    
    return