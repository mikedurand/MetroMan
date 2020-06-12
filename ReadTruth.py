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
    
    fid.close()
    
    return Tru
