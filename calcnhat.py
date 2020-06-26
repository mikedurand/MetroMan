#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:07:09 2020

@author: mtd
"""
from numpy import any

def calcnhat(w,h,hmin,A,x1,na,nOpt):

    if nOpt==3:
        nhat=na * (h-hmin+0.1)**x1
    elif nOpt==4:
        nhat=na * (A/w)**x1    
    elif nOpt==5:
        # this is based on Rodriguez et al. WRR 2020 and assumes a log-normal distribution of river depth
        nhat=na * (1 + (x1/(A/w))**2 )**(5/6)
    
    return nhat
    