from numpy import *
from numpy.linalg import inv
from ManningCalcs import *
from ChannelMassBal import *

def CalcLklhd(h,A0,h0,S,n,w,L,t,sigS,sigh,sigq,dA,Delta,DeltaA,qhatv,Qvbar):

    # All vectors ordered "space-first"
    # theta(1)=theta(r1,t1)
    # theta(2)=theta(r1,t2)
    # ... 
    # theta(nt)=theta(r1,nt)
    # theta(nt+1)=theta(r2,t1)


    [nR,nt]=h.shape
    n=n*ones([nR,1])
    hv=h.reshape(nR*nt,1)
    dAv=dA.reshape(nR*nt,1)
    nv=(n*ones([1,nt])).reshape(nR*nt,1)
    wv=w.reshape(nR*nt,1)
    Sv=S.reshape(nR*nt,1)
    
    A0v=(A0*ones([1,nt])).reshape(nR*nt,1)
    h0v=(h0.T*ones([1,nt])).reshape(nR*nt,1)

    Q=ManningCalcs(A0.T,n,S,dA,w);
    Qv=Q.reshape(nR*nt,1);

    if (hv<0).any() | (A0v<0).any() | (Sv<0).any():
        f=0
        return
    
    # Calculate dQdx, dQdt, and q for floodplain mass balance

    N=nR*(nt-1); # total number of "equations" / constraints
    M=nR*nt;    

#    [dQdx,dAdt]= FloodplainMassBal(h,t,L,Q,dA,F,Delta)
    [dQdx,dAdt]= ChannelMassBal(h,t,L,Q,dA,Delta)
    dQdxv=dQdx.reshape(N,1)

    # Handle Jacobian of slope
    TSv=Sv**(-1)
    if isnan(Qvbar).any():
        JS=0.5*Delta*(ones([N,1])*Qv.T)*(ones([N,1])*TSv.T)
    else:
        JS=0.5*Delta*(ones([N,1])*Qvbar.T)*(ones([N,1])*TSv.T)    
    Cs=sigS**2*eye(M,M,0,float)

    # Handle Jacobian of height
    Thv=wv/(A0v+dAv)
    if isnan(Qvbar).any():
        Jh=(5./3.)*Delta*(ones([N,1])*Qv.T)*(ones([N,1])*Thv.T)
    else:
        Jh=(5./3.)*Delta*(ones([N,1])*Qvbar.T)*(ones([N,1])*Thv.T)
    Ch=sigh**2*eye(M,M,0,float);

    J=concatenate((JS,Jh),axis=1)

    Calpha=concatenate((concatenate((Cs,zeros([M,M],float)),axis=1),\
                       concatenate((zeros([M,M],float),Ch),axis=1)))
    Cq=dot(dot(J,Calpha),J.T)

    # 
    # for r=1:nR,
    #     for j=1:nt-1,
    #         t1=j; t2=j+1;
    #         dt=(t(t2)-t(t1))*86400;
    #         wbar=(w(r,t1)+w(r,t2))/2;
    #         sig2c(r,j)=(wbar/dt)^2*2*sigh^2;
    #     end
    # end

    dt=diff(t).T*86400*ones([1,nR]); dt=dt.reshape([nR*(nt-1),1])
    JAh=(ones([N,1])*wv.T)*DeltaA/(dt*ones([1,M]))
    CA=dot(dot(JAh,Ch),JAh.T)

    # Cc=diag(reshape(sig2c',N,1));
    Cqf=eye(N,N,0,float)*(sigq**2)

    Cf=CA+Cq+Cqf
    # Cf=CA+Cq;

    dAdtv=dAdt.reshape([N,1])
    # qv=reshape(q,N,1);
    # Cf=diag(reshape(sig2f',N,1));

    # f=mvnpdf(dQdxv+dAdtv-qhatv, [], Cf);
    Theta=dQdxv+dAdtv-qhatv

    f=(-0.5)*dot(dot(Theta.T,inv(Cf)),Theta)
    
    return f
