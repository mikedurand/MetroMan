def GetABC(r,nR,L):
    if r==0:
        a='nan'
        b=2/(L[r]+L[r+1])
        c=2/(L[r]+L[r+1])
    elif r==(nR-1):
        a=2/(L[r]+L[r-1])
        b='nan'
        c=-2/(L[r]+L[r-1])  # this - to agree with notation in the paper   
    else:
        a=1/(L[r]+L[r-1])
        b=1/(L[r]+L[r+1])
        c=(L[r-1]-L[r+1])/( (L[r]+L[r-1])*(L[r]+L[r+1]) )
    return a,b,c
