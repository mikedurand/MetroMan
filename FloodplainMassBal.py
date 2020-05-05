# script to calculate mass balance of the channel + floodplain, assuming all
# change in mass on the floodplain is due to the channel

import sys
from numpy import *
from scipy.interpolate import interp1d

def FloodplainMassBal(hin,t,L,Q,dA,F,Delta):
    # Floodplain Mass-balance: assume that all change in Floodplain volume is
    # added back to the channel (i.e. perpendicular flow assumption)

    if len(sys.argv)>7:
        CalcUnc=true
        sig2Q=sys.argv[7]
    else:
        CalcUnc=False
        sig2dQdx=nan

    [nr,nt]=hin.shape
    
    # Interpolate floodplain volumes & areas at each height
    Vf=empty([nr,nt])
    for r in range(0,nr):
        f=interp1d(F.h,F.V[:,r],bounds_error=False); Vf[r,:]=f(hin[r,:])
    
    # Calculate dQdx
    Qv=Q.reshape(nr*nt,1)
    dQdxv=dot(Delta,Qv)
    dQdx=dQdxv.reshape(nr,nt-1)
    sig2dQdx=empty([len(dQdxv),len(dQdxv)]); sig2dQdx[:]=nan

    # Channel mass-balance calculations
    dAdt=empty([nr,nt-1])
    for r in range(0,nr):
        for i in range(0,nt-1):
            t1=i; t2=i+1
            dt=86400*(t[0,t2]-t[0,t1])
            dAdt[r,i]=(dA[r,t2]-dA[r,t1])/dt # channel mass bal

    # Calculate change in floodplain volumes
    dVf=diff(Vf)

    dVfdt=empty([nr,nt-1]); q=empty([nr,nt-1])
    for r in range(0,nr):
        dVfdt[r,:]=dVf[r,:]/diff(t)/86400-dAdt[r,:]*L[r]
        q[r,:]=-dVfdt[r,:]/L[r]
    

    # Calculations for floodplain flow
    VfQ=empty([nr,nt]); AfQ=empty([nr,nt])
    for r in range(0,nr):
        f=interp1d(F.h,F.V[:,r],bounds_error=False)
        VfQ[r,:]=maximum((f(hin[r,:])-f(0.5*(F.hb[r]+F.hb[r+1]))),0)
        f=interp1d(F.h,F.A[:,r],bounds_error=False); AfQ[r,:]=f(hin[r,:])

    Yf=VfQ/AfQ;
    Tf=AfQ/(L.reshape(len(L),1)*ones([1,nt]))

    return dQdx,dAdt
