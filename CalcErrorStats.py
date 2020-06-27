#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:21:06 2020

@author: mtd
"""

from numpy import mean,sqrt,log,std
from MetroManVariables import ErrorStats

def CalcErrorStats(AllTruth,E,DAll):
    
    Qt=mean(AllTruth.Q,axis=0)
    QhatAvg=mean(E.AllQ,axis=0)
    
    Stats=ErrorStats()
    
    Stats.RMSE=sqrt(mean( (Qt-QhatAvg)**2 ) )
    Stats.rRMSE=sqrt(mean( (  (Qt-QhatAvg)/Qt   )**2 ) )
    Stats.nRMSE=Stats.RMSE/mean(Qt)

    print('Discharge RMSE: %.6f' %Stats.RMSE)
    print('Discharge nRMSE: %.6f' %Stats.nRMSE)    
    
    r=QhatAvg-Qt
    logr=log(QhatAvg)-log(Qt)

    Stats.NSE=1-sum(r**2)/sum( (Qt-mean(Qt))**2 )
    Stats.VE=1- sum(abs(r))/sum(Qt)

    Stats.bias=mean(r)
    Stats.stdresid=std(r)
    Stats.nbias = Stats.bias/mean(Qt)

    Stats.MSC=log(  sum((Qt-mean(Qt))**2)/sum(r**2) -2*2 / DAll.nt  )
    Stats.meanLogRes=mean(logr)
    Stats.stdLogRes=std(logr)
    Stats.meanRelRes=mean(r/Qt)
    Stats.stdRelRes=std(r/Qt)

    Stats.Qbart=mean(Qt)  
    Stats.Stats=Stats
    
    return Stats