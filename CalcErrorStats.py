#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:21:06 2020

@author: mtd
"""

from numpy import mean,sqrt
from MetroManVariables import ErrorStats

def CalcErrorStats(AllTruth,E,DAll):
    
    Qt=mean(AllTruth.Q,axis=0)
    QhatAvg=mean(E.AllQ,axis=0)
    
    Stats=ErrorStats()
    
    Stats.RMSE=sqrt(mean( (Qt-QhatAvg)**2 ) )
    Stats.nRMSE=Stats.RMSE/mean(Qt)

    print('Discharge RMSE: %.6f' %Stats.RMSE)
    print('Discharge nRMSE: %.6f' %Stats.nRMSE)    
    
    return Stats