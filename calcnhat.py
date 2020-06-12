#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:07:09 2020

@author: mtd
"""


def calcnhat(w,h,hmin,A,x1,na,nOpt):

    if nOpt==3:
        nhat=na * (h-hmin+0.1)**x1
    elif nOpt==4:
        nhat=na * (A/w)**x1    
    
    return nhat
    