#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:59:05 2020

@author: mtd
"""

from numpy import log,sqrt

def logninvstat(m,v):
    
    mu = log((m**2)/sqrt(v+m**2));
    sigma = sqrt(log(v/(m**2)+1));
    
    return mu,sigma