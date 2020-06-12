#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:58:17 2020

@author: mtd
"""

from numpy import mean
import matplotlib.pyplot as plt

def MakeFigs(D,Truth,Prior,C,Estimate,Err,AllTruth,DAll,AllObs):
    
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
    
    plt.figure(2)
    plt.hist(C.thetana[0,C.Nburn+1:C.N],bins=100)
    plt.title('Reach 0, Mean = %.4f' %(mean(C.thetana[0,C.Nburn+1:C.N])) )
    
    plt.figure(3)
    plt.plot(DAll.t.T,Estimate.AllQ[0,:].reshape((DAll.nt,1)),label='estimate' ) 
    plt.plot(DAll.t.T,AllTruth.Q[0,:].reshape((DAll.nt,1)),label='true' )
    plt.legend()
    plt.xlabel('time,days')
    plt.ylabel('discharge m^3/s')
    
    return