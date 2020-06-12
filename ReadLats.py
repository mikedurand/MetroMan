#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import zeros,empty,array

def ReadLats(fname,D):
    
    fid=open(fname,"r")
    infile=fid.readlines()
    
    q=empty( (D.nR,D.nt-1) ) 
    
    for i in range(0,D.nR):        
        buf=infile[i+1]; buf=buf.split(); q[i,:]=array(buf,float)
    
    return q