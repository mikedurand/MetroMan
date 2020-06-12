#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:20:46 2020

@author: mtd
"""

from numpy import zeros,empty

def CalcB(nR,nt):
    
    Row=-1 
    
    M=nR * nt
    N=nR *(nt-1)    
    
    B=zeros((N,M))
    
    for r in range(0,nR):
        for i in range(0,nt-1):
            Row=Row+1
            
            # B[Row,:]=zeros((M,0))
            
            t1=i; t2=i+1;
            Col1=t1+r*nt
            Col2=t2+r*nt
            
            B[Row,Col1]=0.5
            B[Row,Col2]=0.5
    
    return B