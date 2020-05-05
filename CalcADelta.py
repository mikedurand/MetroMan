from numpy import empty,nan,zeros

def CalcADelta(nr,nt):
    Row=-1

    N=nr*(nt-1)
    M=nr*nt
    Delta=empty([N,M]); Delta[:]=nan

    for r in range(0,nr):
        for i in range(0,nt-1):
            Row=Row+1;
            Delta[Row,:]=zeros([1,M])

            t1=i; t2=i+1
            Col1=t1+r*nt
            Col2=t2+r*nt

            Delta[Row,Col1]=-1.0;
            Delta[Row,Col2]=1.0;

    return Delta
