#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array,diff,ones,reshape
from MetroManVariables import Domain,Observations

#%%
def ReadObs(fname):

    fid=open(fname,"r")
    infile=fid.readlines()
    
    # read domain
    D=Domain()
    D.nR=eval(infile[1])
    buf=infile[3]; buf=buf.split(); D.xkm=array(buf,float)
    buf=infile[5]; buf=buf.split(); D.L=array(buf,float)
    D.nt=eval(infile[7]);
    buf=infile[9]; buf=buf.split(); D.t=array([buf],float)
    
    D.dt=reshape(diff(D.t).T*86400 * ones((1,D.nR)),(D.nR*(D.nt-1),1))
    
    
    #%% read observations   
    Obs=Observations(D)
    for i in range(0,D.nR):
        buf=infile[i+11]; buf=buf.split(); Obs.h[i,:]=array(buf,float)
    buf=infile[12+D.nR]; buf=buf.split(); Obs.h0=array([buf],float)
    for i in range(0,D.nR):
        buf=infile[14+D.nR+i]; buf=buf.split(); Obs.S[i,:]=array(buf,float)/1e5; #convert cm/km -> m/m
    for i in range(0,D.nR):
        buf=infile[15+D.nR*2+i]; buf=buf.split(); Obs.w[i,:]=array(buf,float)
    Obs.sigS=eval(infile[16+D.nR*3])/1e5; #convert cm/km -> m/m
    Obs.sigh=eval(infile[18+D.nR*3])/1e2; #convert cm -> m
    Obs.sigw=eval(infile[20+D.nR*3] )
        
    #%% create resahepd versions of observations
    Obs.hv=reshape(Obs.h, (D.nR*D.nt,1) )
    Obs.Sv=reshape(Obs.S, (D.nR*D.nt,1) )
    Obs.wv=reshape(Obs.w, (D.nR*D.nt,1) )
    
    #%%
    fid.close()   

    return D,Obs