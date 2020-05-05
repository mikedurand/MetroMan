# script to calculate mass balance of the channel + floodplain, assuming all
# change in mass on the floodplain is due to the channel

import sys
from numpy import *
from scipy.interpolate import interp1d

def ChannelMassBal(hin,t,L,Q,dA,Delta):
    # Floodplain Mass-balance: assume that all change in Floodplain volume is
    # added back to the channel (i.e. perpendicular flow assumption)

    if len(sys.argv)>7:
        CalcUnc=true
        sig2Q=sys.argv[7]
    else:
        CalcUnc=False
        sig2dQdx=nan

    [nr,nt]=hin.shape
    
    # Calculate dQdx
    Qv=Q.reshape(nr*nt,1)
    dQdxv=dot(Delta,Qv)
    dQdx=dQdxv.reshape(nr,nt-1)

    # Channel mass-balance calculations
    dAdt=empty([nr,nt-1])
    for r in range(0,nr):
        for i in range(0,nt-1):
            t1=i; t2=i+1
            dt=86400*(t[0,t2]-t[0,t1])
            dAdt[r,i]=(dA[r,t2]-dA[r,t1])/dt # channel mass bal

    return dQdx,dAdt
