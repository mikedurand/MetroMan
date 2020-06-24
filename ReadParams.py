#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array
from MetroManVariables import Chain,RandomSeeds,Experiment,Prior

def ReadParams(fname,D):
    fid=open(fname,"r")
    infile=fid.readlines()

    C=Chain()
    C.N=eval(infile[1])
    C.Nburn=eval(infile[3])
    
    R=RandomSeeds()
    R.Seed=eval(infile[5])
    
    Exp=Experiment()
    buf=infile[7]; buf=buf.split(); Exp.tUse=array(buf,float)
    
    P=Prior(D)
    P.meanQbar=eval(infile[9])
    P.covQbar=eval(infile[11])
    
    Exp.nOpt=eval(infile[13])
    #Exp.tStep=eval(infile[15])
    
    if len(infile)>14:
        buf=infile[15]; buf=buf.split();
        P.Geomorph.Use=True
        P.Geomorph.loga_hat=eval(buf[0])
        P.Geomorph.loga_sigma=eval(buf[1])
        P.Geomorph.b_hat=eval(buf[2])
        P.Geomorph.b_sigma = eval(buf[3])
        P.Geomorph.logA0_hat = eval(buf[4])
        P.Geomorph.logA0_sigma = eval(buf[5])
    else:
        P.Geomorph.Use=False
            
    
    
    return C,P,R,Exp
        