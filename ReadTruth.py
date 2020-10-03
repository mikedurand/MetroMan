#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array
from MetroManVariables import Truth

def ReadTruth(fname,D):
    
    fid=open(fname,"r")
    infile=fid.readlines()
    
    Tru=Truth(D)
    
    buf=infile[1]; buf=buf.split(); Tru.A0=array(buf,float)
    buf=infile[3]; Tru.q=buf #not fully implemented; only affects plotting routines
    buf=infile[5]; Tru.n=buf #not fully implemented; only affects plotting routines
    
    for i in range(0,D.nR):
        buf=infile[i+7]; buf=buf.split(); Tru.Q[i,:]=array(buf,float) 
        buf_dA=infile[i+8+D.nR]; buf_dA=buf_dA.split(); Tru.dA[i,:]=array(buf_dA,float)
        buf_h=infile[i+9+2*D.nR]; buf_h=buf_h.split(); Tru.h[i,:]=array(buf_h,float)
        buf_W=infile[i+10+3*D.nR]; buf_W=buf_W.split(); Tru.W[i,:]=array(buf_W,float)
        
    
    fid.close()
    
    return Tru
