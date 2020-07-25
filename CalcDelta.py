from numpy import empty,nan,zeros
from GetABC import GetABC

def CalcDelta(nr,nt,L):
    Row=-1

    N=nr*(nt-1)
    M=nr*nt
    Delta=empty([N,M]); Delta[:]=nan

    for r in range(0,nr):
        [a,b,c]=GetABC(r,nr,L)

        for i in range(0,nt-1):
            t1=i; t2=i+1
            Row=Row+1

            Delta[Row,:]=zeros([1,M])

            if r==0:
                Col1=t1
                Col2=t2
                Col3=t1+nt
                Col4=t2+nt
                Delta[Row,Col1:Col2+1]=-c/2
                Delta[Row,Col3:Col4+1]=b/2
            elif r==(nr-1):
                Col1=t1+nt*(r-1)
                Col2=t2+nt*(r-1)
                Col3=t1+nt*(r)
                Col4=t2+nt*(r)
                Delta[Row,Col1:Col2+1]=-a/2
                Delta[Row,Col3:Col4+1]=-c/2
            else:
                Col1=t1+nt*(r-1)
                Col2=t2+nt*(r-1)
                Col3=t1+nt*(r)
                Col4=t2+nt*(r)
                Col5=t1+nt*(r+1)
                Col6=t2+nt*(r+1)
                Delta[Row,Col1:Col2+1]=-a/2
                Delta[Row,Col3:Col4+1]=-c/2
                Delta[Row,Col5:Col6+1]=b/2
                     
    return Delta
