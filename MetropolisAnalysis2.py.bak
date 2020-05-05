# script to estimate A0 given real h & S data: standalone
# alpha development

import time
import scipy.io as sio
from numpy import *
from numpy.random import *
from CalcDelta import *
from CalcADelta import *
from CalcLklhd import *

# read June data
fid=open("tst/JuneData.txt","r")
infile=fid.readlines()

nR=eval(infile[1])
buf=infile[3]; buf=buf.split(); xkm=array(buf,float)
buf=infile[5]; buf=buf.split(); L=array(buf,float)
nt=eval(infile[7]);
buf=infile[9]; buf=buf.split(); t=array([buf],float)
hin=empty([nR,nt])
for i in range(0,nR):
    buf=infile[i+11]; buf=buf.split(); hin[i,:]=array(buf,float)
buf=infile[15]; buf=buf.split(); h0bar=array([buf],float)
Sin=empty([nR,nt])
for i in range(0,nR):
    buf=infile[i+17]; buf=buf.split(); Sin[i,:]=array(buf,float);
Sin=Sin/1e5
dA=empty([nR,nt])
for i in range(0,nR):
    buf=infile[i+21]; buf=buf.split(); dA[i,:]=array(buf,float)
wobs=empty([nR,nt])
for i in range(0,nR):
    buf=infile[i+25]; buf=buf.split(); wobs[i,:]=array(buf,float)
fid.close()   
    
# read June parameters
fid=open("tst/JuneParams.txt","r")
infile=fid.readlines()

N=eval(infile[1])
Nburn=eval(infile[3])
buf=infile[5]; buf=buf.split(); meanA0=array([buf],float); 
buf=infile[7]; buf=buf.split(); stdA0=array(buf,float)
meann=eval(infile[9])
stdn=eval(infile[11])
buf=infile[13]; buf=buf.split(); meanq=array([buf],float)
stdq=eval(infile[15])
sigS=eval(infile[17])/1e5
sigh=eval(infile[19])/1e2

class Jump:
    def __init__(self,stdA0burn,stdA0sim,A0min,stdn,nmin,stdq,qmin):
        self.stdA0burn=stdA0burn
        self.stdA0sim=stdA0sim
        self.A0min=A0min
        self.stdn=stdn
        self.nmin=nmin
        self.stdq=stdq
        self.qmin=qmin
    def setStdA0(self,stdA0):
        self.stdA0=stdA0   

stdA0burn=eval(infile[21]); stdA0sim=eval(infile[23]); 
A0min=eval(infile[25]); stdn1=eval(infile[27]);
nmin=eval(infile[29]); stdq1=eval(infile[31]); qmin=eval(infile[33]);

jmp=Jump(stdA0burn,stdA0sim,A0min,stdn1,nmin,stdq1,qmin)
del stdA0burn,stdA0sim,A0min,stdn1,nmin,stdq1,qmin
fid.close()

# read Truth
fid=open("tst/Truth.txt","r")
infile=fid.readlines()
buf=infile[1]; buf=buf.split(); A0true=array([buf],float)
buf=infile[3]; buf=buf.split(); qtrue=array(buf,float)
fid.close()

# thins to change later...
# L=L.reshape(1,len(L))
# h0bar=h0bar.reshape(1,len(h0bar))
# t=t.reshape(1,len(t))
# A0true=A0true.reshape(1,len(A0true))

#3) Bookkeeping
# 3.1) allocations, and initial state set
thetaA0=empty([nR,N]); thetaA0[:]=nan # this is just A0
thetaA0[:,0]=meanA0
thetauA0=thetaA0[:,0].copy(); thetauA0=thetauA0.reshape(len(thetauA0),1)

thetan=empty([1,N]); thetan[:]=nan
thetan[:,0]=meann
thetaun=thetan[:,0].copy()

thetaq=empty([nR*(nt-1),N]); thetaq[:]=nan
thetaq[:,0]=meanq;
thetauq=thetaq[:,0].copy(); thetauq=thetauq.reshape(len(thetauq),1)

# 3.2) random numbers used in chain operation
seed([1]) # seed the pseudo-random number generator

z1=randn(nR,N) # used for jumping the layer thickness
z2=randn(1,N)
z3=randn(nR*(nt-1),N); 

u1=rand(N,1); # used for acceptance of A0
u2=rand(N,1); # used for acceptance of n
u3=rand(N,1); # used for acceptance of q

n_a1=0.0; n_a2=0.0; n_a3=0.0

# 3.4) 
Delta = CalcDelta(nR,nt,L);
DeltaA = CalcADelta(nR,nt);

# 4) Metropolis calculations
# 6.1) initial probability calculations
pu1=exp(dot(dot(-0.5*(thetauA0-meanA0.T).T,diagflat(stdA0**-2.)), 
        (thetauA0-meanA0.T)))
pu2=exp(-0.5*(thetaun-meann)**2./stdn**2.)
pu3=exp(dot(-0.5*(thetauq-meanq.T).T*diagflat(stdq**-2.),
        (thetauq-meanq.T)))

Qvbar=empty([nR*nt,1]); Qvbar[:]=nan

fu= CalcLklhd(hin,thetauA0,h0bar,Sin,thetaun,wobs,L,t,sigS,sigh,\
              stdq,dA,Delta,DeltaA,thetauq,Qvbar)

# 6.2) The loop
tic=time.clock()

jmp.setStdA0(jmp.stdA0burn)
Like=empty([1,N])
for i in range(0,N):
    if i%1000==0:
        print "Iteration #", i, "/", N, "."
    if i==Nburn:
        jmp.setStdA0(jmp.stdA0sim)
    
    thetavA0=thetauA0+(jmp.stdA0*z1[:,i]).reshape(len(z1[:,i]),1)
    thetavA0[thetavA0<jmp.A0min]=jmp.A0min
    pv1=exp(dot(dot(-0.5*(thetavA0-meanA0.T).T,diagflat(stdA0**-2.)),(thetavA0-meanA0.T)))
    fv=CalcLklhd(hin,thetavA0,h0bar,Sin,thetaun,wobs,L,t,sigS,sigh,stdq,dA,Delta,DeltaA,thetauq,Qvbar)

    MetRatio=exp(fv-fu)*pv1/pu1
    if MetRatio>u1[i]:
        n_a1=n_a1+1 # increment
        thetauA0=thetavA0; fu=fv;pu1=pv1 # update u->v
    thetaA0[:,i]=thetauA0.T

    # n
    thetavn=thetaun+jmp.stdn*z2[:,i]
    thetavn[thetavn<jmp.nmin]=jmp.nmin
    pv2=exp(-0.5*(thetavn-meann)**2./(stdn**2.))    
    fv=CalcLklhd(hin,thetauA0,h0bar,Sin,thetavn,wobs,L,t,sigS,sigh,stdq,dA,Delta,DeltaA,thetauq,Qvbar)
    
    MetRatio=exp(fv-fu)*pv2/pu2
    if MetRatio>u2[i]:
        n_a2=n_a2+1 # increment
        thetaun=thetavn; fu=fv;pu2=pv2 # update u->v
    thetan[:,i]=thetaun.T
    
    # q
    thetavq=thetauq+(jmp.stdq*z3[:,i]).reshape(len(z3[:,i]),1)
    thetavq[thetavq<jmp.qmin]=jmp.qmin
    pv3=exp(dot(dot(-0.5*(thetavq-meanq.T),diagflat(stdq**-2.)).T,(thetavq-meanq.T))) 
    fv=CalcLklhd(hin,thetauA0,h0bar,Sin,thetaun,wobs,L,t,sigS,sigh,stdq,dA,Delta,DeltaA,thetavq,Qvbar)

    MetRatio=exp(fv-fu)*pv3/pu3
    if MetRatio>u3[i]:
        n_a3=n_a3+1 # increment
        thetauq=thetavq; fu=fv;pu3=pv3 # update u->v
    thetaq[:,i]=thetauq.T

    Like[0,i]=exp(fu)
toc=time.clock(); print 'Computing Time: %.2fs' %(toc-tic)

print 'A0: Acceptance rate =',(n_a1/N*100), ' pct.'
print 'n: Acceptance rate =', (n_a2/N*100), ' pct.'
print 'q: Acceptance rate =', (n_a3/N*100), ' pct.'

A0hat=mean(thetaA0[:,Nburn+1:N],1)
CA0=cov(thetaA0[:,Nburn+1:N])
stdA0Post=sqrt(diagonal(CA0))

nhat=mean(thetan[0,Nburn+1:N])
stdnPost=std(thetan[0,Nburn+1:N])

qhat=mean(thetaq[:,Nburn+1:N],1)
Cq=cov(thetaq[:,Nburn+1:N])
stdqpost=sqrt(diagonal(Cq))

# save with structure (est)
est={'A0hat':A0hat,'stdA0Post':stdA0Post,'nhat':nhat,'stdnPost':stdnPost,'qhat':qhat,\
     'stdqpost':stdqpost,'jmp':jmp,'N':N,'n_a1':n_a1,'n_a2':n_a2,'n_a3':n_a3}
sio.savemat('Estimate.mat',{'est':est},oned_as='row')

# plot
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(311)
plt.plot(thetaA0.T)
plt.title('Baseflow cross-sectional area, m^2')
plt.subplot(312)
plt.plot(thetan.T)
plt.title('Roughness coefficient')
plt.subplot(313)
plt.plot(mean(thetaq.T,1))
plt.title('Average q, m2/s')

plt.figure(2);
p1=plt.errorbar(xkm/1000,A0hat,stdA0Post,linewidth=2,color='r'); plt.hold(True)
p2=plt.errorbar(xkm/1000,meanA0.reshape(size(meanA0),),stdA0,linewidth=2,color='g')
p3=plt.errorbar(xkm/1000,A0true.T,linewidth=2); plt.hold(False)
plt.xlabel('Flow distance, km')
plt.ylabel('Cross-sectional area, m^2')
plt.legend([p1,p2,p3],['Estimate','Prior','True'],3)     

plt.figure(3)
for i in range(0,nR):
    plt.subplot(1,3,i+1)
    plt.hist(thetaA0[i,Nburn+1:N],bins=100)

plt.figure(4)
plt.hist(thetan[0,Nburn+1:N],bins=100)
plt.title('Mean = %.4f' %(mean(thetan[0,Nburn+1:N])))

qhatm=qhat.reshape(nR,nt-1).T
qtruem=qtrue.reshape(nR,nt-1).T

plt.figure(5)
plt.plot(t[0,1:nt],qhatm); plt.hold(True)
plt.plot(t[0,1:nt],qtruem,'--'); plt.hold(False)

# Quick Error stats
RelErrA0=mean(A0hat.T-A0true)/mean(A0true)
RelErrN=(nhat-0.035)/0.035
RMSq=sqrt(mean((qhat-qtrue)**2.0))
RMSqPrior=sqrt(mean((meanq-qtrue)**2.))

print 'Relative Error in A0: %.6f' %RelErrA0
print 'Relative Uncertainty in A0: %.6f' %mean(stdA0Post.T/A0hat.T)

print 'Relative Error in n: %.6f' %RelErrN
print 'Relative Uncertainty in n: %.6f' %(stdnPost/nhat)

print 'RMS for q posterior: %.6f' %RMSq
print 'Relative Uncertainty in q: %.6f' %mean(stdqpost/qhat)

plt.figure(6)
plt.plot(mean(thetaA0[:,Nburn+1:N].T,1),Like[0,Nburn+1:N].T,'.')

plt.figure(7)
plt.plot(range(1,N+1),Like.T)

plt.figure(8)
for i in range(0,nR*(nt-1)):
    plt.subplot(3,6,i+1)
    plt.hist(thetaq[i,Nburn+1:N]); plt.hold(True)
    a=plt.axis();
    plt.plot(array([meanq[0,i],meanq[0,i]])-stdq,array([a[2],a[3]]))
    plt.plot(array([meanq[0,i],meanq[0,i]])+stdq,array([a[2],a[3]]))
    plt.plot([qtrue[i],qtrue[i]],[a[2],a[3]],'-',color=[0,0.5,0])
    plt.plot([meanq[0,i],meanq[0,i]],[a[2],a[3]],'r:'); 
    plt.plot(mean(thetaq[i,Nburn+1:N]),a[3],'bs')
    plt.hold(False)

plt.figure(9)
plt.plot(qtrue,qhat,'bo',qtrue,meanq.reshape(size(meanq),),'rx',[.2e-3,2e-3],[.2e-3,2e-3])

from matplotlib.mlab import normpdf

plt.figure(10)
plt.subplot(121)
x=arange(1,501)
y=normpdf(x,mean(meanA0),mean(stdA0));
plt.plot(x,y)
plt.title('1D prior on A0 Mean = %.0f' %meanA0[0,0])
plt.subplot(122)
x=arange(.01,0.051,.001)
y=normpdf(x,meann,stdn);
plt.plot(x,y)
plt.title('1D prior on n Mean = %.2f' %meann )

plt.show()
