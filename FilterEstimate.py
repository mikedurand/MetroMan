"""
Module for filtering estimate
"""

from numpy import empty,NaN,diag,var,linalg,mean,eye,sqrt
from CalcDelta import CalcDelta
from CalcADelta import CalcADelta
from CalcB import CalcB

def FilterEstimate(Estimate,C,D,Obs):
    
    Deltax = CalcDelta(D.nR,D.nt,D.L)
    DeltaA = CalcADelta(D.nR,D.nt)
    B = CalcB(D.nR,D.nt)

    H=-Deltax
    y=(DeltaA@Obs.hv) / D.dt * (B@Obs.wv)
    
    Qchain=empty([D.nR*D.nt,C.N])
    Qchain[:]=NaN
    for i in range(0,C.N):
        Qchain[:,i]=C.thetaQ[i,:,:].reshape(D.nR*D.nt)    
    
    P=diag(var(Qchain,1).reshape(D.nR*D.nt))
    R=Obs.CA

    K=P@H.T @ linalg.inv((H@P@H.T+R))
    
    xminus=empty([D.nR*D.nt,C.N]); xminus[:]=NaN
    xplus=empty([D.nR*D.nt,C.N]); xplus[:]=NaN
    for i in range(0,C.N):
        xminus[:,i]=Qchain[:,i]    
        xplus[:,i]=xminus[:,i] + (K@(y-H@xminus[:,i].reshape([D.nR*D.nt,1]))).reshape(D.nR*D.nt)
    

    Qhatfv=mean(xplus[:,C.Nburn-1:C.N-1],1)
    Estimate.QhatPostf=Qhatfv.reshape([D.nR,D.nt])    
    Ppostv=diag( (eye(D.nt*D.nR)-K@H)@P )
    Estimate.QhatPostfUnc=sqrt(Ppostv).reshape([D.nR,D.nt])
    return Estimate